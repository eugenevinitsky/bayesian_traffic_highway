from typing import Dict, Tuple

from gym.envs.registration import register
import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv, MultiAgentWrapper
from highway_env.envs.intersection_env import IntersectionEnv
from highway_env.road.lane import LineType, StraightLane, CircularLane, AbstractLane
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import RoadNetwork
from highway_env.vehicle.kinematics import Vehicle

from highway_env.vehicle.controller import MDPVehicle, ControlledVehicle
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.l012vehicles import L0Vehicle, L1Vehicle, L2Vehicle, Pedestrian, FullStop

# 3 sources of truth: 2x here and 1x in l012vehicles.py, because I can't reference the env from the vehicles (could pass the env obj in ...)


NUM_NON_EGO_VEHICLES = 3
CAR_FEATURE_NAMES = ["x", "y", "vx", "vy", "heading", "arr_order"]
EGO_FEATURE_NAMES = [f"ego_{feature}" for feature in CAR_FEATURE_NAMES]
PED_FEATURE_NAMES = ["ped_0", "ped_1", "ped_2", "ped_3"]
NON_EGO_FEATURE_NAMES = [f"non_ego_{i}_{feature}" for feature in CAR_FEATURE_NAMES for i in range(NUM_NON_EGO_VEHICLES)]

class PedestrianIntersectionEnv(IntersectionEnv):

    def __init__(self, config):
        super().__init__(config)
        self.arrival_position = dict()
        
    @classmethod
    def default_config(cls) -> dict:
        print('default config called')
        config = super().default_config()
        config.update({
            "observation": {
                "type": "IntersectionWithPedObservation",
                "num_non_ego_vehicles": NUM_NON_EGO_VEHICLES,
                "car_feature_names": CAR_FEATURE_NAMES,
                "ego_feature_names": EGO_FEATURE_NAMES,
                "ped_feature_names": PED_FEATURE_NAMES,
                "non_ego_feature_names": NON_EGO_FEATURE_NAMES,
                "feature_names": EGO_FEATURE_NAMES + PED_FEATURE_NAMES + NON_EGO_FEATURE_NAMES,
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20],
                },
                "absolute": True,
                "flatten": False,
                "observe_intentions": False
            },
            "action": {
                "type": "ContinuousAction",
                "longitudinal": True,
                "acceleration_range": [-1, 1],  # only matters for L2
                "lateral": False # ah, here's where we turn things off
            },
            "scenario": None,
            "inference_noise_std": 0.0,
            "duration": 13,  # [s]
            "destination": "o1",
            "controlled_vehicles": 1,
            "initial_vehicle_count": 10,
            "spawn_probability": 0.6,
            "screen_width": 600,
            "screen_height": 600,
            "centering_position": [0.5, 0.6],
            "scaling": 5.5 * 0.6,
            "collision_reward": IntersectionEnv.COLLISION_REWARD,
            "normalize_reward": False,
            "simulation_frequency": 10,
            "policy_frequency": 10
        })
        return config

    def _make_road(self) -> None:
        """
        Make an 4-way intersection.

        The horizontal road has the right of way. More precisely, the levels of priority are:
            - 3 for horizontal straight lanes and right-turns
            - 1 for vertical straight lanes and right-turns
            - 2 for horizontal left-turns
            - 0 for vertical left-turns

        The code for nodes in the road network is:
        (o:outer | i:inner + [r:right, l:left]) + (0:south | 1:west | 2:north | 3:east)

        :return: the intersection road
        """
        lane_width = AbstractLane.DEFAULT_WIDTH
        right_turn_radius = lane_width  # + 5   # [m}
        left_turn_radius = right_turn_radius + lane_width  # [m}
        outer_distance = right_turn_radius + lane_width / 2   # 7.5
        access_length = 50 + 50  # [m]

        net = RoadNetwork()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED

        self.config["true_scenario"] = self.config["scenario"]
        if self.config["scenario"] == 11:
            self.config["true_scenario"] = np.random.choice([1, 2, 9, 10])
            self.config["train"] = True
            self.config["train_noise"] = np.random.uniform()
            print(f'scenario: {self.config["scenario"]}')

        if self.config["true_scenario"] in [9, 10]:
            # 2nd lane for this scenario (and 3rd invisible lane)
            # incoming from south
            rotation = np.array([[1, 0], [0, 1]])
            start = rotation @ np.array([3 * lane_width / 2, access_length + outer_distance])
            end = rotation @ np.array([3 * lane_width / 2, outer_distance + lane_width])
            net.add_lane("o0", "ir0",
                         StraightLane(start, end, line_types=[s, c], priority=0, speed_limit=10, forbidden=True))
            if self.config["true_scenario"] == 10:
                start = rotation @ np.array([5 * lane_width / 2, access_length + outer_distance])
                end = rotation @ np.array([5 * lane_width / 2, outer_distance + lane_width])
                net.add_lane("o0", "ir0",
                            StraightLane(start, end, line_types=[n, n], priority=0, speed_limit=10, forbidden=True))
            # intersection
            r_center = rotation @ (np.array([outer_distance + lane_width, outer_distance + lane_width]))
            net.add_lane("ir0", "il3",
                         CircularLane(r_center, right_turn_radius, 0 + np.radians(180), 0 + np.radians(270),
                                      line_types=[n, c], priority=0, speed_limit=10))
            # outgoing east
            start = rotation @ np.flip([3 * lane_width / 2, access_length + outer_distance], axis=0)
            end = rotation @ np.flip([3 * lane_width / 2, outer_distance + lane_width], axis=0)
            net.add_lane("il3", "o3",
                         StraightLane(end, start, line_types=[s, c], priority=0, speed_limit=10))

        for corner in range(4):
            angle = np.radians(90 * corner)
            is_horizontal = corner % 2
            priority = 3 if is_horizontal else 1
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            # Incoming
            lt = [c, s] if self.config["true_scenario"] in [9, 10] and corner == 0 else [s, c]
            start = rotation @ np.array([lane_width / 2, access_length + outer_distance])
            end = rotation @ np.array([lane_width / 2, outer_distance])
            net.add_lane("o" + str(corner), "ir" + str(corner),
                         StraightLane(start, end, line_types=lt, priority=priority, speed_limit=10, forbidden=True))

            # replace s by n to hide ped lanes
            ped_display = s
            if self.config["true_scenario"] in [9]:
                # Pedestrian lanes (other direction and further from intersection)
                start = rotation @ np.array([lane_width * 2.5, outer_distance * 2])
                end = rotation @ np.array([lane_width * 2.5, -outer_distance * 2])
                net.add_lane("p" + str(corner), "p" + str(corner) + "_end",
                            StraightLane(end, start, line_types=[ped_display, ped_display], priority=1234,
                                        width=AbstractLane.DEFAULT_WIDTH / 2, speed_limit=2))
            elif self.config["true_scenario"] in [10]:
                # Pedestrian lanes (other direction and further from intersection)
                start = rotation @ np.array([lane_width * 8, outer_distance * 2.5])
                end = rotation @ np.array([lane_width * 8, -outer_distance * 2.5])
                net.add_lane("p" + str(corner), "p" + str(corner) + "_end",
                            StraightLane(end, start, line_types=[ped_display, ped_display], priority=1234,
                                        width=AbstractLane.DEFAULT_WIDTH / 2, speed_limit=2))
            else:
                # Pedestrian lanes
                start = rotation @ np.array([lane_width * 1.5,  outer_distance * 2])
                end = rotation @ np.array([lane_width * 1.5, -outer_distance * 2])
                net.add_lane("p" + str(corner), "p" + str(corner) + "_end",
                            StraightLane(start, end, line_types=[ped_display, ped_display], priority=1234,
                                        width=AbstractLane.DEFAULT_WIDTH / 2, speed_limit=2))
                
            # Right turn
            lt = [c, s] if self.config["true_scenario"] in [9, 10] and corner == 0 else [n, c]
            r_center = rotation @ (np.array([outer_distance, outer_distance]))
            net.add_lane("ir" + str(corner), "il" + str((corner - 1) % 4),
                         CircularLane(r_center, right_turn_radius, angle + np.radians(180), angle + np.radians(270),
                                      line_types=[n, c], priority=priority, speed_limit=10))
            # Left turn
            l_center = rotation @ (np.array([-left_turn_radius + lane_width / 2, left_turn_radius - lane_width / 2]))
            net.add_lane("ir" + str(corner), "il" + str((corner + 1) % 4),
                         CircularLane(l_center, left_turn_radius, angle + np.radians(0), angle + np.radians(-90),
                                      clockwise=False, line_types=[n, n], priority=priority - 1, speed_limit=10))
            # Straight
            start = rotation @ np.array([lane_width / 2, outer_distance])
            end = rotation @ np.array([lane_width / 2, -outer_distance])
            net.add_lane("ir" + str(corner), "il" + str((corner + 2) % 4),
                         StraightLane(start, end, line_types=[s, n], priority=priority, speed_limit=10))
            # Exit
            lt = [c, s] if self.config["true_scenario"] in [9, 10] and corner == 0 else [s, c]
            start = rotation @ np.flip([lane_width / 2, access_length + outer_distance], axis=0)
            end = rotation @ np.flip([lane_width / 2, outer_distance], axis=0)
            net.add_lane("il" + str((corner - 1) % 4), "o" + str((corner - 1) % 4),
                         StraightLane(end, start, line_types=lt, priority=priority, speed_limit=10))

        road = RegulatedRoad(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _reset(self) -> None:
        super()._reset()
        self.set_observer_vehicle('L0Vehicle')

    def set_observer_vehicle(self, veh_name):
        """Set observer vehicle to vehicle with name veh_name
        
        @Parameters
        veh_type: string
        """
        for v in self.road.vehicles:
            if 'L0Vehicle' in str(type(v)):
                self.observation_type.set_observer_vehicle(v)

    def reset_priors(self):
        for v in self.road.vehicles:
            if isinstance(v, L1Vehicle):
                v.priors = dict()

    def _make_vehicles(self, n_vehicles: int = 10) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane

        :return: the ego-vehicle
        """
        # Configure vehicles
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle_type.DISTANCE_WANTED = 7  # Low jam distance
        vehicle_type.COMFORT_ACC_MAX = 2.6
        vehicle_type.COMFORT_ACC_MIN = -4.5

        self.controlled_vehicles = []
        
        def spawn_vehicle(lane, dest, pos, speed, scenario, heading=None, type="car", vclass=L0Vehicle, controlled=False):
            params = {
                ## we adapt the dynamics depending on what scenario is used
                'scenario': scenario,
                'inference_noise_std': self.config['inference_noise_std'],
                # distance of the pedestrian crossing from the origin
                'crossing_positions': [(4, 10), (-10, 0), (0, -10), (10, 0)] if scenario == 9 \
                    else [(4, 32), (-32, 0), (0, -32), (32, 0)] if scenario == 10 \
                    else [(0, 6), (-6, 0), (0, -6), (6, 0)],
                # distance to detect crossings
                'crossing_distance_detection': 7 if scenario == 9 else 4,
                # safe margin for vehicles to stop before crossings
                'safe_margin': 8 if scenario in [9, 10] \
                    else 10 if scenario in [2] \
                    else 10,
                # max x distance pedestrians can be detected at (useful for vehicles traveling vertically)
                # for scenario 10 we don't want L0 on left lane to stop if the pedestrian is on the right side of the road
                'max_ped_detection_x': 2 if scenario in [10] else 9999,
                # pedestrian is deleted once this condition is true
                'ped_delete_condition': (lambda ped: ped.position[1] > 7) if scenario in [1] \
                    else (lambda ped: ped.position[0] < -3) if scenario in [9] \
                    else (lambda ped: ped.position[0] < 0) if scenario in [10] \
                    else lambda ped: False,
                'env': self
            }
            vehicle = vclass.make_on_lane(self.road, lane, longitudinal=pos, speed=speed, params=params)

            if vclass == L0Vehicle:
                vehicle.color = (190, 190, 190)
            elif vclass == L1Vehicle:
                vehicle.color = (230, 80, 220)
            elif vclass == L2Vehicle:
                vehicle.color = (65, 150, 240)
            elif vclass == FullStop:
                vehicle.color = (0, 0, 0)
            elif vclass == Pedestrian:
                vehicle.color = (255, 255, 255)

            if type == "bus":
                vehicle.WIDTH = 3
                vehicle.LENGTH = 10
                vehicle.color = (240, 190, 70)
            elif type == "car":
                vehicle.WIDTH = 3
                vehicle.LENGTH = 6
            elif type == "ped":
                vehicle.WIDTH = 2
                vehicle.LENGTH = 2
            elif type == "tree":
                vehicle.WIDTH = 4
                vehicle.LENGTH = 4
                vehicle.color = (0, 255, 0)

            if heading:
                vehicle.heading = heading
            vehicle.plan_route_to(dest)
            vehicle.randomize_behavior()

            if controlled:
                self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)


        def scenario_1(train=False, noise=0):
            spawn_vehicle(scenario=1, vclass=L0Vehicle, lane=("o1", "ir1", 0), dest="o3", pos=70 + noise * 5, speed=8.0 + noise * 2, type="car", controlled=True)
            spawn_vehicle(scenario=1, vclass=L1Vehicle if not train else L0Vehicle, lane=("o3", "ir3", 0), dest="o1", pos=75 + noise * 5, speed=14 + noise * 2, type="car")
            spawn_vehicle(scenario=1, vclass=Pedestrian, lane=("p2", "p2_end", 0), dest="p2_end", pos=2, speed=2.0, type="ped")
            spawn_vehicle(scenario=1, vclass=FullStop, lane=("o2", "ir2", 0), dest="o1", pos=97, speed=0.0, type="bus")
            # spawn_vehicle(vclass=FullStop, lane=("ir2", "il1", 0), dest="o1", pos=2, speed=2.0, heading=3.1415/16*12, type="bus")
        def scenario_2(train=False, noise=0):
            spawn_vehicle(scenario=2, vclass=L1Vehicle if not train else L0Vehicle, lane=("o3", "ir3", 0), dest="o1", pos=60 + noise * 5, speed=9.0 + noise * 2, type="car")
            spawn_vehicle(scenario=2, vclass=L2Vehicle if not train else L0Vehicle, lane=("o1", "ir1", 0), dest="o0", pos=80 + noise * 5, speed=9.0 + noise * 2, type="car", controlled=True)
            spawn_vehicle(scenario=2, vclass=FullStop, lane=("o2", "ir2", 0), dest="o0", pos=95 + noise * 5, speed=0.0, type="bus")
            spawn_vehicle(scenario=2, vclass=FullStop, lane=("il3", "o3", 0), dest="o3", pos=8 + noise * 5, speed=0.0, type="bus")
            spawn_vehicle(scenario=2, vclass=Pedestrian, lane=("p0", "p0_end", 0), dest="p0_end", pos=2, speed=2.0, type="ped")
            # spawn_vehicle(vclass=Pedestrian, lane=("p2", "p2_end", 0), dest="p2_end", pos=0, speed=0.0, type="ped")
        def scenario_9(train=False, noise=0):
            spawn_vehicle(scenario=9, vclass=L1Vehicle if not train else L0Vehicle, lane=("o0", "ir0", 1), dest="o2", pos=40, speed=9.0 + noise * 2, type="car", controlled=True)
            spawn_vehicle(scenario=9, vclass=L0Vehicle, lane=("o0", "ir0", 0), dest="o3", pos=60 + noise * 1, speed=8.0 + noise * 1, type="car")
            spawn_vehicle(scenario=9, vclass=L0Vehicle, lane=("o0", "ir0", 0), dest="o3", pos=50 + noise * 1, speed=8.0 + noise * 1, type="car")
            spawn_vehicle(scenario=9, vclass=L0Vehicle, lane=("o0", "ir0", 0), dest="o3", pos=40 + noise * 1, speed=8.0 + noise * 1, type="car")
            spawn_vehicle(scenario=9, vclass=Pedestrian, lane=("p1", "p1_end", 0), dest="p1_end", pos=-3, speed=2.0, type="ped")        
        def scenario_10(train=False, noise=0):
            spawn_vehicle(scenario=10, vclass=L1Vehicle if not train else L0Vehicle, lane=("o0", "ir0", 0), dest="o3", pos=30 + noise * 2, speed=9.0 + noise * 1, type="car")
            spawn_vehicle(scenario=10, vclass=L2Vehicle if not train else L0Vehicle, lane=("o0", "ir0", 2), dest="o2", pos=40, speed=9.0 + noise * 1, type="car", controlled=True)
            spawn_vehicle(scenario=10, vclass=FullStop, lane=("o0", "ir0", 1), dest="o3", pos=70, speed=0.0, type="tree")
            spawn_vehicle(scenario=10, vclass=Pedestrian, lane=("p1", "p1_end", 0), dest="p1_end", pos=-1, speed=2.0, type="ped")
        def scenario_11():
            print(f'scenario 11 called\nnn\n\n')
            n1 = np.random.uniform()
            n2 = np.random.uniform()
            scenario_10(train=True, noise=n2)
        # TODO KL get the scenario things correct
            # if 0 <= n1 < 0.25:
            #     scenario_1(l0=True, noise=n2)
            # elif 0.25 <= n1 < 0.5:
            #     scenario_2(l0=True, noise=n2)
            # elif 0.5 <= n1 < 0.75:
            #     scenario_9(l0=True, noise=n2)
            # elif 0.75 <= n1 <= 1:
            #     scenario_10(l0=True, noise=n2)
        train = self.config.get("train", False) 
        train_noise = self.config.get("train_noise", 0)
        if self.config["scenario"] is not None:
            print(f"Using custom scenario: {self.config['scenario']}")
            if self.config["true_scenario"] == 0: # debug
                raise ValueError
            elif self.config["true_scenario"] == 1:
                scenario_1(train=train, noise=train_noise)
            elif self.config["true_scenario"] == 2:
                scenario_2(train=train, noise=train_noise)
            elif self.config["true_scenario"] == 9:
                scenario_9(train=train, noise=train_noise)
            elif self.config["true_scenario"] == 10:
                scenario_10(train=train, noise=train_noise)
            else:
                raise ValueError(f"Scenario '{self.config['true_scenario']}' unknown.")

            for v in self.road.vehicles:
                if not isinstance(v, Pedestrian) and not isinstance(v, FullStop):
                    v.precompute()
            return

        # Controlled vehicles
        self.controlled_vehicles = []
        for ego_id in range(0, self.config["controlled_vehicles"]):
            ego_lane = self.road.network.get_lane(("o{}".format(ego_id % 4), "ir{}".format(ego_id % 4), 0))
            destination = self.config["destination"] or "o" + str(self.np_random.randint(1, 4))
            ego_vehicle = self.action_type.vehicle_class(
                             self.road,
                             ego_lane.position(60 + 5*self.np_random.randn(1), 0),
                             speed=ego_lane.speed_limit,
                             heading=ego_lane.heading_at(60)) \
                #.plan_route_to(destination)
            ego_vehicle.SPEED_MIN = 0
            ego_vehicle.SPEED_MAX = 9
            ego_vehicle.SPEED_COUNT = 3
            # ego_vehicle.speed_index = ego_vehicle.speed_to_index(ego_lane.speed_limit)
            # ego_vehicle.target_speed = ego_vehicle.index_to_speed(ego_vehicle.speed_index)

            self.road.vehicles.append(ego_vehicle)
            self.controlled_vehicles.append(ego_vehicle)
            for v in self.road.vehicles:  # Prevent early collisions
                if v is not ego_vehicle and np.linalg.norm(v.position - ego_vehicle.position) < 20:
                    self.road.vehicles.remove(v)

    def _spawn_vehicle(self,
                       longitudinal: float = 0,
                       position_deviation: float = 1.,
                       speed_deviation: float = 1.,
                       spawn_probability: float = 0.6,
                       go_straight: bool = False,
                       is_ped: bool = False) -> None:
        if self.np_random.rand() > spawn_probability:
            return

        route = self.np_random.choice(range(4), size=2, replace=False)
        route[1] = (route[0] + 2) % 4 if go_straight else route[1]
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        if is_ped:
            vehicle = vehicle_type.make_on_lane(self.road, ("p" + str(route[0]), "p" + str(route[0]) + "_end", 0),
                                                longitudinal=longitudinal + 5 + self.np_random.randn() * position_deviation,
                                                speed=2.0)
            vehicle.LENGTH = 2.0
            vehicle.color = (200, 0, 150)
        else:
            vehicle = vehicle_type.make_on_lane(self.road, ("o" + str(route[0]), "ir" + str(route[0]), 0),
                                                longitudinal=longitudinal + 5 + self.np_random.randn() * position_deviation,
                                                speed=8 + self.np_random.randn() * speed_deviation)
        for v in self.road.vehicles:
            if np.linalg.norm(v.position - vehicle.position) < 15:
                return
        vehicle.plan_route_to("o" + str(route[1]))
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)
        return vehicle

register(
    id='intersection-pedestrian-v0',
    entry_point='highway_env.envs:PedestrianIntersectionEnv',
)
