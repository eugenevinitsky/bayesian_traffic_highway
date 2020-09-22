from typing import List, Tuple

import numpy as np

from highway_env import utils
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import StraightLane
from highway_env.vehicle.controller import ControlledVehicle, MDPVehicle
from highway_env.vehicle.kinematics import Vehicle, Obstacle


class RegulatedRoad(Road):
    YIELDING_COLOR: Tuple[float, float, float] = None
    REGULATION_FREQUENCY: int = 2
    YIELD_DURATION: float = 0.

    def __init__(self, network: RoadNetwork = None, vehicles: List[Vehicle] = None, obstacles: List[Obstacle] = None,
                 np_random: np.random.RandomState = None, record_history: bool = False) -> None:
        super().__init__(network, vehicles, obstacles, np_random, record_history)
        self.steps = 0

        self.intersection_order = []

    def step(self, dt: float) -> None:
        self.steps += 1
        if self.steps % int(1 / dt / self.REGULATION_FREQUENCY) == 0:
            self.enforce_road_rules()
        return super().step(dt)

    def enforce_road_rules(self) -> None:
        """Find conflicts and resolve them by assigning yielding vehicles and stopping them."""

        print('step', self.steps)

        for corner in range(4):
            print('ped crossing', corner, self.is_ped_crossing(corner))

        if False:  # go one by one in intersection
            n_veh_in_intersection = 0
            closest = None
            closest_dist = 1e9
            threshold = 10  # threshold to start braking if theres someone in intersection
            threshold_arrival = 50 # threshold to be detected as arriving to intersection

            # prevent vehicles from entering intersection if there's already one vehicle in
            for v in self.vehicles:
                origin, dest, _ = lane = self.network.get_closest_lane_index(v.position)
                lane_geometry = self.network.get_lane(lane)

                if origin.startswith("o"):
                    # going towards intersection
                    assert isinstance(lane_geometry, StraightLane)
                    _, lane_end = lane_geometry.start, lane_geometry.end
                    v_dist = np.linalg.norm(lane_end - v.position)  # distance to intersection
                    if v_dist < closest_dist:
                        closest_dist = v_dist
                        closest = v
                    if v_dist < threshold:
                        # vehicle getting close to intersection -> brake
                        v.target_speed = 0
                    if v_dist < threshold_arrival:
                        if v not in self.intersection_order:
                            self.intersection_order.append(v)
                elif origin.startswith("i") and dest.startswith("i"):
                    # vehicle in intersection
                    n_veh_in_intersection += 1
            
            if n_veh_in_intersection == 0 and len(self.intersection_order) > 0: # and closest is not None:
                # if intersection is empty, last vehicle arrived is allowed to go
                v_go = self.intersection_order.pop(0)
                v_go.target_speed = v_go.lane.speed_limit

                # # if intersection is empty, vehicle closest to intersection is allowed to go
                # closest.target_speed = closest.lane.speed_limit


        # Unfreeze previous yielding vehicles
        for v in self.vehicles:
            if getattr(v, "is_yielding", False):
                if v.yield_timer >= self.YIELD_DURATION * self.REGULATION_FREQUENCY:
                    v.target_speed = v.lane.speed_limit
                    # if hasattr(v, "color"):
                    #     delattr(v, "color")
                    v.is_yielding = False
                else:
                    v.yield_timer += 1

        # Find new conflicts and resolve them
        for i in range(len(self.vehicles) - 1):
            for j in range(i+1, len(self.vehicles)):
                # avoid crashes
                if self.is_conflict_possible(self.vehicles[i], self.vehicles[j]):
                    yielding_vehicle = self.respect_priorities(self.vehicles[i], self.vehicles[j])
                    if yielding_vehicle is not None and \
                            isinstance(yielding_vehicle, ControlledVehicle) and \
                            not isinstance(yielding_vehicle, MDPVehicle):
                        # yielding_vehicle.color = self.YIELDING_COLOR
                        yielding_vehicle.target_speed = 0
                        yielding_vehicle.is_yielding = True
                        yielding_vehicle.yield_timer = 0

                # if ped is crossing, slow down
                self.check_ped_crossing(self.vehicles[i], self.vehicles[j])

    @staticmethod
    def respect_priorities(v1: Vehicle, v2: Vehicle) -> Vehicle:
        """
        Resolve a conflict between two vehicles by determining who should yield

        :param v1: first vehicle
        :param v2: second vehicle
        :return: the yielding vehicle
        """
        if v1.lane.priority > v2.lane.priority:
            return v2
        elif v1.lane.priority < v2.lane.priority:
            return v1
        else:  # The vehicle behind should yield
            return v1 if v1.front_distance_to(v2) > v2.front_distance_to(v1) else v2

    @staticmethod
    def is_conflict_possible(v1: ControlledVehicle, v2: ControlledVehicle, horizon: int = 3, step: float = 0.25) -> bool:
        times = np.arange(step, horizon, step)
        positions_1, headings_1 = v1.predict_trajectory_constant_speed(times)
        positions_2, headings_2 = v2.predict_trajectory_constant_speed(times)

        for position_1, heading_1, position_2, heading_2 in zip(positions_1, headings_1, positions_2, headings_2):
            # Fast spherical pre-check
            if np.linalg.norm(position_2 - position_1) > v1.LENGTH:
                continue

            # Accurate rectangular check
            if utils.rotated_rectangles_intersect((position_1, 1.5*v1.LENGTH, 0.9*v1.WIDTH, heading_1),
                                                  (position_2, 1.5*v2.LENGTH, 0.9*v2.WIDTH, heading_2)):
                return True

    @staticmethod
    def check_ped_crossing(v1: ControlledVehicle, v2: ControlledVehicle) -> bool:
        times_veh = np.arange(0.1, 1, 0.1)
        times_ped = np.arange(0.1, 5, 0.1)

        if v1.LENGTH < 3 and v2.LENGTH > 3:
            ped, veh = v1, v2
        elif v1.LENGTH > 3 and v2.LENGTH < 3:
            ped, veh = v2, v1
        else:
            return False
 
        for pos1 in ped.predict_trajectory_constant_speed(times_ped)[0]:
            for pos2 in veh.predict_trajectory_constant_speed(times_veh, use_lane_speed=True)[0]:
                if np.linalg.norm(pos1 - pos2) < 5 and \
                        isinstance(veh, ControlledVehicle) and \
                        not isinstance(veh, MDPVehicle):
                    veh.target_speed = 0
                    veh.is_yielding = True
                    veh.yield_timer = 0
                    return True
        return False

    def is_ped_crossing(self, crossing: int) -> bool:
        """whether a pedestrian is crossing on the specified crossing.
        crossing ids =
            0: east crossing
            1: south crossing
            2: west crossing
            3: north crossing"""
        for v in self.vehicles:
            if type(v.lane) is StraightLane:
                if v.lane_index[0].startswith(f'p{crossing}'):
                    if np.max(v.lane.local_coordinates(v.position)) > 95 and np.max(v.lane.local_coordinates(v.position)) < 115:
                        return True             
        return False