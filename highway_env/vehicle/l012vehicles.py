from typing import Tuple, Union
import numpy as np
from collections import defaultdict

from highway_env.road.road import Road, Route, LaneIndex
from highway_env.types import Vector
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.behavior import IDMVehicle
from highway_env import utils
from highway_env.vehicle.kinematics import Vehicle
from highway_env.road.objects import RoadObject
from highway_env.bayesian_inference.inference import get_filtered_posteriors


# priors for inference
priors = {}

# minimal and maximal acceleration for all vehicles
ACC_MIN = -4
ACC_MAX = 3

class FullStop(IDMVehicle):
    """Vehicle that shouldn't move (eg obstacle)"""
    def acceleration(self, ego_vehicle, front_vehicle=None, rear_vehicle=None, peds=None):    
        return ACC_MIN

class Pedestrian(IDMVehicle):
    """Pedestrian"""
    pass

class L0Vehicle(IDMVehicle):
    """Rule-based vehicle that obeys normal traffic rules"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # L0 can see through obstacles
        self.is_obscured = False
        # position of center points of the 4 crossings
        self.crossings_positions = self.params['crossing_positions']

    def precompute(self):
        # get which crossings the car will cross along its route
        ids_crossed = []
        times = np.arange(0, 20, 0.2)
        for pred_pos in self.predict_trajectory_constant_speed(times, use_lane_speed=True)[0]:
            for crossing_id in range(4):
                cpos = self.crossings_positions[crossing_id]
                if np.abs(cpos[0] - pred_pos[0]) + np.abs(cpos[1] - pred_pos[1]) < self.params['crossing_distance_detection']:
                    ids_crossed.append(crossing_id)
        self.ids_crossed = set(ids_crossed)

    def acceleration(self, ego_vehicle, front_vehicle=None, rear_vehicle=None, peds=None):
        # compute default IDM acceleration (pedestrians are not accounted for as obstacles)
        if isinstance(front_vehicle, Pedestrian): front_vehicle = None
        if isinstance(rear_vehicle, Pedestrian): rear_vehicle = None
        accel = super().acceleration(ego_vehicle, front_vehicle, rear_vehicle)

        # get crossings where there are pedestrians (without obstruction) unless provided
        if not peds:
            assert (not self.is_obscured)  # to make sure there's no bug since L1 and L2 use this method
            peds = [False] * 4  # south, west, north, east
            for v in self.road.vehicles:
                if isinstance(v, Pedestrian):
                    start_edge = v.lane_index[0]
                    # start edge is p<idx> with idx = 0 south, 1 west, 2 north, 3 east
                    if start_edge.startswith("p"):  # and not start_edge.endswith("inv"):
                        if np.abs(v.position[0] - self.position[0]) < self.params['max_ped_detection_x']:
                        # if True: np.abs(v.position[0]) < 8 and np.abs(v.position[1]) < 8:  # if ped is within crossing
                            # crossing ids = 0 east, 1 south, 2 west, 3 north; we want south to be 0, hence the -1 % 4
                            idx = (int(start_edge[1]) - 1) % 4
                            peds[idx] = True
        
        # if a crossing with a ped is on the path, start breaking when we need to
        for idx in range(4):
            if peds[idx] and idx in self.ids_crossed:
                crossing_pos = self.crossings_positions[idx]
                dist_to_crossing = np.abs(self.position[0] - crossing_pos[0]) + np.abs(self.position[1] - crossing_pos[1])
                # get next distance to make sure vehicle is going towards crossing
                next_pos = self.predict_trajectory_constant_speed([0.01], use_lane_speed=True)[0][0]
                next_dist_to_crossing = np.abs(next_pos[0] - crossing_pos[0]) + np.abs(next_pos[1] - crossing_pos[1])
                if next_dist_to_crossing <= dist_to_crossing:
                    # car stops after v^2/2a meters if current speed is v and breaking with constant acceleration -a
                    if dist_to_crossing - self.params['safe_margin'] < self.speed * self.speed / (-2 * ACC_MIN):
                        accel = ACC_MIN

        accel = np.clip(accel, ACC_MIN, ACC_MAX)
        return accel

class L1Vehicle(L0Vehicle):
    """Rule based vehicle (L0) operating on an inferred belief state"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # L1 has obscured view
        self.is_obscured = True
        # whether L1 does inference (set to False to make sure it crashes without inference)
        self.do_inference = True
         # whether L1 can see non-obscured peds (set to False if L1 should use inference only)
        self.see_peds = True

        # save data here for plotting later
        self.plot_data = defaultdict(list)

    def acceleration(self, ego_vehicle, front_vehicle=None, rear_vehicle=None, peds=None):
        # compute visible vehicles
        visible_vehs = self.road.close_vehicles_to(self, distance=50, count=5, see_behind=False,
            obscuration=True, fov=130, looking_distance=50, verbal=False)

        # remove front/rear vehicles if not visible
        if front_vehicle not in visible_vehs: front_vehicle = None
        if rear_vehicle not in visible_vehs: rear_vehicle = None

        # get peds visible by L1
        peds_visible = [v for v in visible_vehs if isinstance(v, Pedestrian)] if self.see_peds else []
        # compute peds vector from visible peds
        peds = [False] * 4
        for ped in peds_visible:
            if ped.lane_index[0].startswith("p"):
                idx = (int(ped.lane_index[0][1]) - 1) % 4  # cf parent method
                peds[idx] = True

        # compute default accel for L1
        accel = super().acceleration(ego_vehicle, front_vehicle, rear_vehicle, peds=peds)

        # compute cars visible by L1, which are inferred as L0 cars
        visible_cars = [v for v in visible_vehs if not isinstance(v, Pedestrian) and not isinstance(v, FullStop)]

        # inference loop
        plotted = False
        if self.do_inference:
            for veh in visible_cars:
                # get action taken by car
                front_v, rear_v = veh.road.neighbour_vehicles(veh)
                if isinstance(veh, L2Vehicle):
                    car_accel = veh.action['acceleration']
                else:
                    car_accel = veh.acceleration(veh, front_v, rear_v)

                # get inferred peds probs and update priors
                updated_ped_probs, priors[veh] = get_filtered_posteriors(
                    veh, car_accel, priors.get(veh, {}),
                    noise_std=self.params['inference_noise_std']
                )

                # plot data
                if isinstance(veh, L0Vehicle):
                    plotted = True
                    self.plot_data['l0_accel'].append(car_accel)
                    self.plot_data['ped_probs'].append(updated_ped_probs)

                # if predicting peds, take action that L0 would take given its knowledge is the prediction
                if np.any(np.array(updated_ped_probs) > 0.8):
                    peds = list(np.array(updated_ped_probs) > 0.8)
                    self_front_v, self_rear_v = self.road.neighbour_vehicles(self)
                    new_accel = super().acceleration(self, self_front_v, self_rear_v, peds=peds)
                    accel = min(new_accel, accel)

        accel = np.clip(accel, ACC_MIN, ACC_MAX)

        if plotted:
            self.plot_data['l1_accel'].append(accel)

        return accel

class L2Vehicle(L0Vehicle):
    """L2 vehicle"""
    def act(self, action=None):
        if action and 'acceleration' in action:
            action['acceleration'] = np.clip(action['acceleration'], ACC_MIN, ACC_MAX)
        Vehicle.act(self, action)

    def accel_no_ped(self):
        # acceleration that L0 would take if there was no pedestrians
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)
        return super().acceleration(self, front_vehicle, rear_vehicle, peds=[False, False, False, False])