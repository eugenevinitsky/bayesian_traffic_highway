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


class PedestrianIntersectionEnv(IntersectionEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 15,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
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
                "type": "DiscreteMetaAction",
                "longitudinal": True,
                "lateral": False
            },
            "scenario": None,
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
            "normalize_reward": False
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
        right_turn_radius = lane_width + 5  # [m}
        left_turn_radius = right_turn_radius + lane_width  # [m}
        outer_distance = right_turn_radius + lane_width / 2
        access_length = 50 + 50  # [m]

        net = RoadNetwork()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        for corner in range(4):
            angle = np.radians(90 * corner)
            is_horizontal = corner % 2
            priority = 3 if is_horizontal else 1
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            # Incoming
            start = rotation @ np.array([lane_width / 2, access_length + outer_distance])
            end = rotation @ np.array([lane_width / 2, outer_distance])
            net.add_lane("o" + str(corner), "ir" + str(corner),
                         StraightLane(start, end, line_types=[s, c], priority=priority, speed_limit=10))

            # Pedestrian lanes
            start = rotation @ np.array([lane_width * 2.5, access_length + outer_distance])
            end = rotation @ np.array([lane_width * 2.5, -outer_distance - access_length])
            net.add_lane("p" + str(corner), "ir" + str(corner),
                         StraightLane(start, end, line_types=[c, c], priority=1000,
                                      width=AbstractLane.DEFAULT_WIDTH / 2, speed_limit=2))

            # Right turn
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
            start = rotation @ np.flip([lane_width / 2, access_length + outer_distance], axis=0)
            end = rotation @ np.flip([lane_width / 2, outer_distance], axis=0)
            net.add_lane("il" + str((corner - 1) % 4), "o" + str((corner - 1) % 4),
                         StraightLane(end, start, line_types=[n, c], priority=priority, speed_limit=10))

        road = RegulatedRoad(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _make_vehicles(self, n_vehicles: int = 10) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane

        :return: the ego-vehicle
        """
        # Configure vehicles
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle_type.DISTANCE_WANTED = 7  # Low jam distance
        vehicle_type.COMFORT_ACC_MAX = 6
        vehicle_type.COMFORT_ACC_MIN = -3

        def spawn_ego(lane, dest, pos):
            # add ego
            ego_lane = self.road.network.get_lane(lane)
            ego_vehicle = self.action_type.vehicle_class(
                            self.road,
                            ego_lane.position(pos, 0),
                            speed=ego_lane.speed_limit,
                            heading=ego_lane.heading_at(60)) \
                .plan_route_to(dest)
            ego_vehicle.SPEED_MIN = 0
            ego_vehicle.SPEED_MAX = 9
            ego_vehicle.SPEED_COUNT = 3
            ego_vehicle.speed_index = ego_vehicle.speed_to_index(ego_lane.speed_limit)
            ego_vehicle.target_speed = ego_vehicle.index_to_speed(ego_vehicle.speed_index)

            self.road.vehicles.append(ego_vehicle)
            self.controlled_vehicles = [ego_vehicle]

        def spawn_vehicle(lane, dest, pos, speed, type="car"):
            vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
            vehicle = vehicle_type.make_on_lane(self.road, lane, longitudinal=pos, speed=speed)
            if type == "ped":
                vehicle.LENGTH = 2.0
                vehicle.color = (200, 0, 150)
            elif type == "bus":
                vehicle.LENGTH = 10.0
                vehicle.color = (200, 210, 0)
            vehicle.plan_route_to(dest)
            vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)

        if self.config["scenario"] is not None:
            print(f"Using custom scenario: {self.config['scenario']}")

            if self.config["scenario"] == "social_sensing":
                spawn_ego(lane=("o3", "ir3", 0), dest="o1", pos=60)
                spawn_vehicle(lane=("o1", "ir1", 0), dest="o3", pos=70, speed=8.0)
                spawn_vehicle(lane=("p2", "ir2", 0), dest="o0", pos=97, speed=2.0, type="ped")
                spawn_vehicle(lane=("o2", "ir2", 0), dest="o1", pos=100, speed=0.0, type="bus")
            else:
                raise ValueError(f"Scenario '{self.config['scenario']}' unknown.")
            return

        # Random vehicles
        simulation_steps = 3
        for t in range(n_vehicles - 1):
            self._spawn_vehicle(np.linspace(0, 80, n_vehicles)[t])
        for _ in range(simulation_steps):
            [(self.road.act(), self.road.step(1 / self.config["simulation_frequency"])) for _ in range(self.config["simulation_frequency"])]

        # Random pedestrians
        simulation_steps = 3
        # TODO(@evinitsky) remove hardcoding
        for t in range(n_vehicles - 1):
            self._spawn_vehicle(80, is_ped=True)
        for _ in range(simulation_steps):
            [(self.road.act(), self.road.step(1 / self.config["simulation_frequency"])) for _ in
             range(self.config["simulation_frequency"])]

        # Challenger vehicle
        self._spawn_vehicle(60, spawn_probability=1, go_straight=True, position_deviation=0.1, speed_deviation=0)

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
                .plan_route_to(destination)
            ego_vehicle.SPEED_MIN = 0
            ego_vehicle.SPEED_MAX = 9
            ego_vehicle.SPEED_COUNT = 3
            ego_vehicle.speed_index = ego_vehicle.speed_to_index(ego_lane.speed_limit)
            ego_vehicle.target_speed = ego_vehicle.index_to_speed(ego_vehicle.speed_index)

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
            vehicle = vehicle_type.make_on_lane(self.road, ("p" + str(route[0]), "ir" + str(route[0]), 0),
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
