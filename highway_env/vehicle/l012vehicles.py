from typing import Tuple, Union
import numpy as np
from collections import defaultdict
import torch
from gym import spaces
import numpy as np
import pandas as pd

from highway_env.road.road import Road, Route, LaneIndex
from highway_env.types import Vector
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.behavior import IDMVehicle
from highway_env import utils
from highway_env.vehicle.kinematics import Vehicle
from highway_env.road.objects import RoadObject
from highway_env.road.lane import StraightLane
from highway_env.bayesian_inference.inference import get_filtered_posteriors
from highway_env import utils
from highway_env.vehicle.imitation_controller.policies.MLP_policy import MLPPolicySL
from highway_env.vehicle.imitation_controller.infrastructure import pytorch_util as ptu
from highway_env.envs.common.observation import IntersectionWithPedObservation  
from highway_env.envs.common.finite_mdp import compute_ttc_grid
from highway_env.road.lane import AbstractLane
from highway_env.vehicle.controller import MDPVehicle

# minimal and maximal acceleration for all vehicles
ACC_MIN = -4
ACC_MAX = 3

class FullStop(IDMVehicle):
    """Vehicle that shouldn't move (eg obstacle)"""
    def acceleration(self, ego_vehicle, front_vehicle=None, rear_vehicle=None, peds=None):    
        return ACC_MIN

class Pedestrian(IDMVehicle):
    """Pedestrian"""
    def step(self, dt):
        super().step(dt)
        if self.params['ped_delete_condition'](self):
            self.road.vehicles.remove(self)

NON_EGO_FEATURES = ["0_x", "0_y", "0_vx", "0_vy", "0_heading", "0_arrival_order",
                        "1_x", "1_y", "1_vx", "1_vy", "1_heading", "1_arrival_order", 
                        "2_x", "2_y", "2_vx", "2_vy", "2_heading", "2_arrival_order"]
class L0Vehicle(IDMVehicle):
    """Rule-based vehicle that obeys normal traffic rules
    
    States for imitation or RL
    
    ego-states: x, y, vx, vy, heading, arrival order 
    ped_states: ped vector of length 4
    non-ego-states: x, y, vx, vy, heading, arrival order 

    Assumption, everyone drives straight
    """


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # L0 can see through obstacles
        self.is_obscured = False
        # position of center points of the 4 crossings
        self.crossings_positions = self.params['crossing_positions']
        self.accel = 0
        self.imitation_policy_path = self.params['imitation_policy_path']
        # import ipdb; ipdb.set_trace()
        if self.imitation_policy_path:
            self.imitation_policy = MLPPolicySL(1,
                                        28,
                                        2,
                                        64,
                                        discrete=False,
                                        learning_rate=5e-3)
            self.imitation_policy.load_state_dict(torch.load('/home/thankyou-always/TODO/research/bayesian_traffic_highway/highway_env/vehicle/data/q1_1_intersection-pedestrian-v0_15-12-2020_23-47-18/policy_itr_0.pt'))   
            self.imitation_policy = self.imitation_policy.mean_net
            obs = torch.zeros(1, 1, 28) # .to(torch.device("cuda"))
            self.imitation_policy(obs)                                        
            print('Done restoring learned policy...')
        # arrival order from our P.O.V
        self.arrival_order = dict()
        # state names
        self.ego_states_names = ["x", "y", "vx", "vy", "heading", "arrival_order"]
        self.ped_states_names = ["ped_0", "ped_1", "ped_2", "ped_3"]
        self.non_ego_states_names = ["x", "y", "vx", "vy", "heading", "arrival_order"]
        self.max_other_vehs = 3
        self.state_names = self.ego_states_names + self.ped_states_names + NON_EGO_FEATURES
        self.features_range = None
        self.clip = True
        self.state = dict()

    def normalize_obs(self, df):
        """
        Normalize the observation values.

        For now, assume that the road is straight along the x axis.
        :param Dataframe df: observation data
        """
        if not self.features_range:
            side_lanes = self.road.network.all_side_lanes(self.lane_index)
            self.features_range = {
                    "x": [-5.0 * MDPVehicle.SPEED_MAX, 5.0 * MDPVehicle.SPEED_MAX],
                    "y": [-AbstractLane.DEFAULT_WIDTH * len(side_lanes), AbstractLane.DEFAULT_WIDTH * len(side_lanes)],
                    "vx": [-2*MDPVehicle.SPEED_MAX, 2*MDPVehicle.SPEED_MAX],
                    "vy": [-2*MDPVehicle.SPEED_MAX, 2*MDPVehicle.SPEED_MAX],
                    "heading": [-2*np.pi, 2*np.pi], # TODO check
                    "arrival_order": [-1, 3] # necessary?
                }
            for i in range(self.max_other_vehs):
                self.features_range.update({
                    f"{i}_x": [-5.0 * MDPVehicle.SPEED_MAX, 5.0 * MDPVehicle.SPEED_MAX],
                    f"{i}_y": [-AbstractLane.DEFAULT_WIDTH * len(side_lanes), AbstractLane.DEFAULT_WIDTH * len(side_lanes)],
                    f"{i}_vx": [-2*MDPVehicle.SPEED_MAX, 2*MDPVehicle.SPEED_MAX],
                    f"{i}_vy": [-2*MDPVehicle.SPEED_MAX, 2*MDPVehicle.SPEED_MAX],
                    f"{i}_heading": [-2*np.pi, 2*np.pi], # TODO check
                    f"{i}_arrival_order": [-1, 3] # necessary?
                })
        for feature, f_range in self.features_range.items():
            if feature in df:
                df[feature] = utils.lmap(df[feature], [f_range[0], f_range[1]], [-1, 1])
                if self.clip:
                    df[feature] = np.clip(df[feature], -1, 1)
        return df

    def to_dict(self, origin_vehicle: "Vehicle" = None, observe_intentions: bool = True) -> dict:
        self.state = self.get_state()
        return self.state

    def arrived_at_intersection(self, v):
        """Arrived at intersection means vehicle v is 2m or less away from intersection"""
        threshold_arrival = 2 # threshold to be detected as arriving to intersection

        origin, dest, _ = lane = self.road.network.get_closest_lane_index(v.position)
        lane_geometry = self.road.network.get_lane(lane)

        if origin.startswith("o"):
            # going towards intersection
            assert isinstance(lane_geometry, StraightLane)
            _, lane_end = lane_geometry.start, lane_geometry.end
            v_dist = np.linalg.norm(lane_end - v.position)  # distance to intersection

            return v_dist < threshold_arrival
        
        return False 

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

    def acceleration(self, ego_vehicle, front_vehicle=None, rear_vehicle=None, peds=None, noise=False, infer=False):
        # compute default IDM acceleration (pedestrians are not accounted for as obstacles)
        if isinstance(front_vehicle, Pedestrian): front_vehicle = None
        if isinstance(rear_vehicle, Pedestrian): rear_vehicle = None

        accel = super().acceleration(ego_vehicle, front_vehicle, rear_vehicle)
        # compute arrival orders
        for v in self.road.vehicles:
            if not (isinstance(v, Pedestrian) or isinstance(v, FullStop)):
                if self.arrived_at_intersection(v) and not self.arrival_order.get(v, False):
                    self.arrival_order[v] = len(self.arrival_order) + 1

        # get crossings where there are pedestrians (without obstruction) unless provided
        peds = self.get_ped_state(peds)

        # if a crossing with a ped is on the path, start breaking when we need to
        for idx in range(4):
            if peds[idx] and idx in self.ids_crossed:
                crossing_pos = self.crossings_positions[idx]
                dist_to_crossing = np.abs(self.position[0] - crossing_pos[0]) + np.abs(self.position[1] - crossing_pos[1])
                # get next distance to make sure vehicle is going towards crossing
                v = self.speed * np.array([np.cos(self.heading), np.sin(self.heading)])
                next_pos = self.position + v * 0.01
                next_dist_to_crossing = np.abs(next_pos[0] - crossing_pos[0]) + np.abs(next_pos[1] - crossing_pos[1])

                if next_dist_to_crossing <= dist_to_crossing: # or next_dist_to_crossing > dist_to_crossing + 10:
                    # car stops after v^2/2a meters if current speed is v and breaking with constant acceleration -a
                    if dist_to_crossing - self.params['safe_margin'] < self.speed * self.speed / (-2 * ACC_MIN):
                        accel = ACC_MIN

                    elif dist_to_crossing - self.params['safe_margin'] < self.speed * self.speed / (-2 * ACC_MIN/2):
                        # for scenario 2, more conservative behavior 
                        if self.params['scenario'] == 2:
                            accel = ACC_MIN / 2

        accel = np.clip(accel, ACC_MIN, ACC_MAX)
        if 'L0' in str(self):
            print(self, 'accel is', accel)
        if not infer:
            self.accel = accel

        # peds = list(map(lambda x: int(x), peds))
        # ego_states = [self.position[0], self.position[1], self.heading, self.speed, self.arrival_order.get(self, -1)]
        
        # non_ego_states = []
        # # compute arrival orders
        # for v in self.road.vehicles:
        #     if not (v == self or (isinstance(v, Pedestrian) or isinstance(v, FullStop))):
        #         v_state = [v.position[0], v.position[1], v.heading, v.speed, v.arrival_order.get(v, -1)]
        #         non_ego_states += v_state

        # self.state = ego_states + peds + non_ego_states
        # extra_states = len(self.ego_states_names) + len(self.ped_states_names) + self.max_other_vehs * len(self.non_ego_states_names) - len(self.state)
        self.state = self.get_state(peds)
        if True:

            if '1_arrival_order' not in self.state.keys():
                print('failed')
            else:
                # Add ego-vehicle
                df = pd.DataFrame.from_records([self.state])[self.state_names]
            if True:
                df = self.normalize_obs(df)

            if '1_arrival_order' not in self.state.keys():
                print('failed')
            else:
                df = df[self.state_names]

            obs = df.values.copy()
            obs = ptu.from_numpy(obs)
            obs = obs.view(1, 1, 28)
            return self.imitation_policy(obs)[0][0][0].detach().numpy()
        return accel

    def get_ped_state(self, peds=None):
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
                            # crossing ids = 0 east, 1 south, 2 west, 3 north; we want south to be 0, hence the -1 % 4
                            idx = (int(start_edge[1]) - 1) % 4
                            peds[idx] = True
        return peds

    def get_state(self, peds=None):
        """Return state vector for current vehicle"""
        ped_states = {}
        non_ego_states = {}
        ego_states = {}


        # fill ego states
        ego_states = {'x': self.position[0], 'y': self.position[1], 'heading': self.heading, 
                      'vx': self.velocity[0], 'vy': self.velocity[1], 'arrival_order': self.arrival_order.get(self, -1)}
        
        # fill ped states
        ped_state_lst = self.get_ped_state(peds)
        for i, ped_val in enumerate(ped_state_lst):
            ped_states[f'ped_{i}'] = int(ped_val)

        # zero-fill non ego states
        for idx in range(self.max_other_vehs):
            non_ego_states[str(idx) + '_' + 'x'] = 0
            non_ego_states[str(idx) + '_' + 'y'] = 0
            non_ego_states[str(idx) + '_' + 'vx'] = 0
            non_ego_states[str(idx) + '_' + 'vy'] = 0
            non_ego_states[str(idx) + '_' + 'heading'] = 0
            non_ego_states[str(idx) + '_' + 'arrival_order'] = -1 

        # fill non ego states
        for idx, v in enumerate(self.road.vehicles):
            if not (v == self or (isinstance(v, Pedestrian) or isinstance(v, FullStop))):
                non_ego_states[str(idx) + '_' + 'x'] = v.position[0]
                non_ego_states[str(idx) + '_' + 'y'] = v.position[1]
                non_ego_states[str(idx) + '_' + 'vx'] = v.velocity[0]
                non_ego_states[str(idx) + '_' + 'vy'] = v.velocity[1]
                non_ego_states[str(idx) + '_' + 'heading'] = v.heading
                non_ego_states[str(idx) + '_' + 'arrival_order'] = v.arrival_order.get(v, -1) 

        self.state.update(ego_states)
        self.state.update(ped_states)
        self.state.update(non_ego_states)

        return self.state

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
        # L1 priors
        self.priors = dict()

    def acceleration(self, ego_vehicle, front_vehicle=None, rear_vehicle=None, peds=None):
        # plot data
        for v in self.road.vehicles:
            if not (isinstance(v, Pedestrian) or isinstance(v, FullStop)):
                self.plot_data[v].append((self.timer, v.speed, v.accel))

        # compute visible vehicles
        visible_vehs = self.road.close_vehicles_to(self, obscuration=True)
        # remove front/rear vehicles if not visible
        if front_vehicle not in visible_vehs: front_vehicle = None
        if rear_vehicle not in visible_vehs: rear_vehicle = None

        # compute cars visible by L1, which are inferred as L0 cars
        visible_cars = [v for v in visible_vehs if not isinstance(v, Pedestrian) and not isinstance(v, FullStop)]

        # get peds visible by L1
        peds_visible = [v for v in visible_vehs if isinstance(v, Pedestrian)] if self.see_peds else []

        # compute peds vector from visible peds
        peds = [False] * 4
        for ped in peds_visible:
            if ped.lane_index[0].startswith("p"):
                idx = (int(ped.lane_index[0][1]) - 1) % 4  # cf parent method
                peds[idx] = True

        # compute default accel for L1
        accel = super().acceleration(ego_vehicle, front_vehicle, None, peds=peds, infer=True)

        # inference loop
        if self.do_inference:
            for veh in visible_cars:
                # get action taken by car
                front_v, rear_v = veh.road.neighbour_vehicles(veh)
                car_accel = veh.acceleration(veh, front_v, rear_v, infer=True)

                # get inferred peds probs and update priors
                updated_ped_probs, self.priors[veh] = get_filtered_posteriors(
                    veh, car_accel, self.priors.get(veh, {}),
                    noise_std=self.params['inference_noise_std']
                )   

                # plot data
                if isinstance(veh, L0Vehicle):
                    self.plot_data['ped_probs'].append(updated_ped_probs)
                    self.plot_data['time'].append(self.timer)

                # if predicting peds, take action that L0 would take given its knowledge is the prediction
                if np.any(np.array(updated_ped_probs) > 0.8):
                    peds = list(np.array(updated_ped_probs) > 0.8)
                    peds[3] = False
                    self_front_v, self_rear_v = self.road.neighbour_vehicles(self)
                    new_accel = super().acceleration(self, self_front_v, self_rear_v, peds=peds, infer=True)
                    accel = min(new_accel, accel)

        accel = np.clip(accel, ACC_MIN, ACC_MAX)

        self.accel = accel
        return accel

class L2Vehicle(L0Vehicle):
    """L2 vehicle"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.see_peds = True
        self.is_obscured = True

    def act(self, action=None):
        if action and 'acceleration' in action:
            self.last_action = action['acceleration']
        super().act(action)

    def acceleration(self, ego_vehicle, front_vehicle=None, rear_vehicle=None, peds=None, noise=False, infer=False):
        # if L1 asks for inference, return what L0 would do
        if peds is not None:
            return super().acceleration(ego_vehicle, front_vehicle, rear_vehicle, peds=peds, infer=True)

        if self.last_action < 0:
            # brake
            accel = ACC_MIN
        else:
            # get acceleration that IDM would output if pedestrians were ignored
            visible_vehs = self.road.close_vehicles_to(self, obscuration=True)

            if front_vehicle not in visible_vehs: front_vehicle = None
            if rear_vehicle not in visible_vehs: rear_vehicle = None

            peds_visible = [v for v in visible_vehs if isinstance(v, Pedestrian)] if self.see_peds else []
            peds = [False] * 4
            for ped in peds_visible:
                if ped.lane_index[0].startswith("p"):
                    idx = (int(ped.lane_index[0][1]) - 1) % 4  # cf parent method
                    peds[idx] = True

            accel = super().acceleration(ego_vehicle, front_vehicle, rear_vehicle, peds=peds, infer=True)

        accel = np.clip(accel, ACC_MIN, ACC_MAX)
        if not infer:
            self.accel = accel
        return accel
