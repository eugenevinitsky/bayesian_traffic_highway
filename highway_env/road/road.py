import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Dict, TYPE_CHECKING, Optional
import math

from highway_env.logger import Loggable
from highway_env.road.lane import LineType, StraightLane, AbstractLane
from highway_env.road.objects import Landmark

if TYPE_CHECKING:
    from highway_env.vehicle import kinematics
    from highway_env.road import objects

logger = logging.getLogger(__name__)

LaneIndex = Tuple[str, str, int]
Route = List[LaneIndex]


class RoadNetwork(object):
    graph: Dict[str, Dict[str, List[AbstractLane]]]

    def __init__(self):
        self.graph = {}

    def add_lane(self, _from: str, _to: str, lane: AbstractLane) -> None:
        """
        A lane is encoded as an edge in the road network.

        :param _from: the node at which the lane starts.
        :param _to: the node at which the lane ends.
        :param AbstractLane lane: the lane geometry.
        """
        if _from not in self.graph:
            self.graph[_from] = {}
        if _to not in self.graph[_from]:
            self.graph[_from][_to] = []
        self.graph[_from][_to].append(lane)

    def get_lane(self, index: LaneIndex) -> AbstractLane:
        """
        Get the lane geometry corresponding to a given index in the road network.

        :param index: a tuple (origin node, destination node, lane id on the road).
        :return: the corresponding lane geometry.
        """
        _from, _to, _id = index
        if _id is None: #and len(self.graph[_from][_to]) == 1:
            _id = 0

        return self.graph[_from][_to][_id]

    def get_closest_lane_index(self, position: np.ndarray) -> LaneIndex:
        """
        Get the index of the lane closest to a world position.

        :param position: a world position [m].
        :return: the index of the closest lane.
        """
        indexes, distances = [], []
        for _from, to_dict in self.graph.items():
            for _to, lanes in to_dict.items():
                for _id, l in enumerate(lanes):
                    distances.append(l.distance(position))
                    indexes.append((_from, _to, _id))
        return indexes[int(np.argmin(distances))]

    def next_lane(self, current_index: LaneIndex, route: Route = None, position: np.ndarray = None,
                  np_random: np.random.RandomState = np.random) -> LaneIndex:
        """
        Get the index of the next lane that should be followed after finishing the current lane.

        - If a plan is available and matches with current lane, follow it.
        - Else, pick next road randomly.
        - If it has the same number of lanes as current road, stay in the same lane.
        - Else, pick next road's closest lane.
        :param current_index: the index of the current lane.
        :param route: the planned route, if any.
        :param position: the vehicle position.
        :param np_random: a source of randomness.
        :return: the index of the next lane to be followed when current lane is finished.
        """
        _from, _to, _id = current_index
        next_to = None
        # Pick next road according to planned route
        if route:
            if route[0][:2] == current_index[:2]:  # We just finished the first step of the route, drop it.
                route.pop(0)
            if route and route[0][0] == _to:  # Next road in route is starting at the end of current road.
                _, next_to, _ = route[0]
            elif route:
                logger.warning("Route {} does not start after current road {}.".format(route[0], current_index))
        # Randomly pick next road
        if not next_to:
            try:
                next_to = list(self.graph[_to].keys())[np_random.randint(len(self.graph[_to]))]
            except KeyError:
                # logger.warning("End of lane reached.")
                return current_index

        # If next road has same number of lane, stay on the same lane
        if len(self.graph[_from][_to]) == len(self.graph[_to][next_to]):
            next_id = _id
        # Else, pick closest lane
        else:
            lanes = range(len(self.graph[_to][next_to]))
            next_id = min(lanes,
                          key=lambda l: self.get_lane((_to, next_to, l)).distance(position))

        return _to, next_to, next_id

    def bfs_paths(self, start: str, goal: str) -> List[List[str]]:
        """
        Breadth-first search of all routes from start to goal.

        :param start: starting node
        :param goal: goal node
        :return: list of paths from start to goal.
        """
        queue = [(start, [start])]
        while queue:
            (node, path) = queue.pop(0)
            if node not in self.graph:
                yield []
            for _next in set(self.graph[node].keys()) - set(path):
                if _next == goal:
                    yield path + [_next]
                elif _next in self.graph:
                    queue.append((_next, path + [_next]))

    def shortest_path(self, start: str, goal: str) -> List[str]:
        """
        Breadth-first search of shortest path from start to goal.

        :param start: starting node
        :param goal: goal node
        :return: shortest path from start to goal.
        """
        return next(self.bfs_paths(start, goal), [])

    def all_side_lanes(self, lane_index: LaneIndex) -> List[LaneIndex]:
        """
        :param lane_index: the index of a lane.
        :return: all lanes belonging to the same road.
        """
        return [(lane_index[0], lane_index[1], i) for i in range(len(self.graph[lane_index[0]][lane_index[1]]))]

    def side_lanes(self, lane_index: LaneIndex) -> List[LaneIndex]:
        """
                :param lane_index: the index of a lane.
                :return: indexes of lanes next to a an input lane, to its right or left.
                """
        _from, _to, _id = lane_index
        lanes = []
        if _id > 0:
            lanes.append((_from, _to, _id - 1))
        if _id < len(self.graph[_from][_to]) - 1:
            lanes.append((_from, _to, _id + 1))
        return lanes

    @staticmethod
    def is_same_road(lane_index_1: LaneIndex, lane_index_2: LaneIndex, same_lane: bool = False) -> bool:
        """Is lane 1 in the same road as lane 2?"""
        return lane_index_1[:2] == lane_index_2[:2] and (not same_lane or lane_index_1[2] == lane_index_2[2])

    @staticmethod
    def is_leading_to_road(lane_index_1: LaneIndex, lane_index_2: LaneIndex, same_lane: bool = False) -> bool:
        """Is lane 1 leading to of lane 2?"""
        return lane_index_1[1] == lane_index_2[0] and (not same_lane or lane_index_1[2] == lane_index_2[2])

    def is_connected_road(self, lane_index_1: LaneIndex, lane_index_2: LaneIndex, route: Route = None,
                          same_lane: bool = False, depth: int = 0) -> bool:
        """
        Is the lane 2 leading to a road within lane 1's route?

        Vehicles on these lanes must be considered for collisions.
        :param lane_index_1: origin lane
        :param lane_index_2: target lane
        :param route: route from origin lane, if any
        :param same_lane: compare lane id
        :param depth: search depth from lane 1 along its route
        :return: whether the roads are connected
        """
        if RoadNetwork.is_same_road(lane_index_2, lane_index_1, same_lane) \
                or RoadNetwork.is_leading_to_road(lane_index_2, lane_index_1, same_lane):
            return True
        if depth > 0:
            if route and route[0][:2] == lane_index_1[:2]:
                # Route is starting at current road, skip it
                return self.is_connected_road(lane_index_1, lane_index_2, route[1:], same_lane, depth)
            elif route and route[0][0] == lane_index_1[1]:
                # Route is continuing from current road, follow it
                return self.is_connected_road(route[0], lane_index_2, route[1:], same_lane, depth - 1)
            else:
                # Recursively search all roads at intersection
                _from, _to, _id = lane_index_1
                return any([self.is_connected_road((_to, l1_to, _id), lane_index_2, route, same_lane, depth - 1)
                            for l1_to in self.graph.get(_to, {}).keys()])
        return False

    def lanes_list(self) -> List[AbstractLane]:
        return [lane for to in self.graph.values() for ids in to.values() for lane in ids]

    @staticmethod
    def straight_road_network(lanes: int = 4, length: float = 10000, angle: float = 0) -> 'RoadNetwork':
        net = RoadNetwork()
        for lane in range(lanes):
            origin = np.array([0, lane * StraightLane.DEFAULT_WIDTH])
            end = np.array([length, lane * StraightLane.DEFAULT_WIDTH])
            rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
            origin = rotation @ origin
            end = rotation @ end
            line_types = [LineType.CONTINUOUS_LINE if lane == 0 else LineType.STRIPED,
                          LineType.CONTINUOUS_LINE if lane == lanes - 1 else LineType.NONE]
            net.add_lane("0", "1", StraightLane(origin, end, line_types=line_types))
        return net

    def position_heading_along_route(self, route: Route, longitudinal: float, lateral: float) \
            -> Tuple[np.ndarray, float]:
        """
        Get the absolute position and heading along a route composed of several lanes at some local coordinates.

        :param route: a planned route, list of lane indexes
        :param longitudinal: longitudinal position
        :param lateral: : lateral position
        :return: position, heading
        """
        while len(route) > 1 and longitudinal > self.get_lane(route[0]).length:
            longitudinal -= self.get_lane(route[0]).length
            route = route[1:]
        return self.get_lane(route[0]).position(longitudinal, lateral), self.get_lane(route[0]).heading_at(longitudinal)


class Road(Loggable):

    """A road is a set of lanes, and a set of vehicles driving on these lanes."""

    def __init__(self,
                 network: RoadNetwork = None,
                 vehicles: List['kinematics.Vehicle'] = None,
                 road_objects: List['objects.RoadObject'] = None,
                 np_random: np.random.RandomState = None,
                 record_history: bool = False) -> None:
        """
        New road.

        :param network: the road network describing the lanes
        :param vehicles: the vehicles driving on the road
        :param road_objects: the objects on the road including obstacles and landmarks
        :param np.random.RandomState np_random: a random number generator for vehicle behaviour
        :param record_history: whether the recent trajectories of vehicles should be recorded for display
        """
        self.network = network
        self.vehicles = vehicles or []
        self.objects = road_objects or []
        self.np_random = np_random if np_random else np.random.RandomState()
        self.record_history = record_history

    def close_vehicles_to(self, vehicle: 'kinematics.Vehicle', distance: float, count: int = None,
                          see_behind: bool = True,
                          obscuration=True, fov=180, looking_distance=50, verbal=False) -> object:
        if obscuration:
            vehicles = [v for v in self.vehicles
                        if v is not vehicle
                        and self.observed(vehicle.position, math.degrees(vehicle.heading), v.position, fov=fov, looking_distance=looking_distance)]
        else:
            vehicles = [v for v in self.vehicles
                        if np.linalg.norm(v.position - vehicle.position) < distance
                        and v is not vehicle
                        and (see_behind or -2 * vehicle.LENGTH < vehicle.lane_distance_to(v))]

        vehicles = sorted(vehicles, key=lambda v: abs(vehicle.lane_distance_to(v)))

        if obscuration:
            blocked = {v: self.get_blocked_segments(
                vehicle.position,
                v.position,
                math.degrees(v.heading),
                v.LENGTH, v.WIDTH
            ) for v in vehicles}

            hidden_vehicles = [v for v in vehicles if self.check_blocked(vehicle.position, v.position, blocked, v)]  # if needed
            vehicles = [v for v in vehicles if v not in hidden_vehicles]  # visible vehicles

            if verbal:  # verbal print hidden & visible vehicles for debug
                f1 = lambda x: x.LENGTH
                f2 = lambda l: {2: 'pedestrian', 6: 'car', 10: 'bus'}[l]
                print(f'visible: {list(map(f2, map(f1, vehicles)))}, hidden: {list(map(f2, map(f1, hidden_vehicles)))}')

        if count:
            vehicles = vehicles[:count]

        return vehicles

    def act(self) -> None:
        """Decide the actions of each entity on the road."""
        for vehicle in self.vehicles:
            vehicle.act()

    def step(self, dt: float) -> None:
        """
        Step the dynamics of each entity on the road.

        :param dt: timestep [s]
        """
        for vehicle in self.vehicles:
            vehicle.step(dt)
        for vehicle in self.vehicles:
            for other in self.vehicles:
                vehicle.check_collision(other)
            for other in self.objects:
                vehicle.check_collision(other)

    def neighbour_vehicles(self, vehicle: 'kinematics.Vehicle', lane_index: LaneIndex = None) \
            -> Tuple[Optional['kinematics.Vehicle'], Optional['kinematics.Vehicle']]:
        """
        Find the preceding and following vehicles of a given vehicle.

        :param vehicle: the vehicle whose neighbours must be found
        :param lane_index: the lane on which to look for preceding and following vehicles.
                     It doesn't have to be the current vehicle lane but can also be another lane, in which case the
                     vehicle is projected on it considering its local coordinates in the lane.
        :return: its preceding vehicle, its following vehicle
        """
        lane_index = lane_index or vehicle.lane_index
        if not lane_index:
            return None, None
        lane = self.network.get_lane(lane_index)
        s = self.network.get_lane(lane_index).local_coordinates(vehicle.position)[0]
        s_front = s_rear = None
        v_front = v_rear = None
        for v in self.vehicles + self.objects:
            if v is not vehicle and not isinstance(v, Landmark):  # self.network.is_connected_road(v.lane_index,
                # lane_index, same_lane=True):
                s_v, lat_v = lane.local_coordinates(v.position)
                if not lane.on_lane(v.position, s_v, lat_v, margin=1):
                    continue
                if s <= s_v and (s_front is None or s_v <= s_front):
                    s_front = s_v
                    v_front = v
                if s_v < s and (s_rear is None or s_v > s_rear):
                    s_rear = s_v
                    v_rear = v
        return v_front, v_rear

    def dump(self) -> None:
        """Dump the data of all entities on the road."""
        for v in self.vehicles:
            v.dump()

    def get_log(self) -> pd.DataFrame:
        """
        Concatenate the logs of all entities on the road.

        :return: the concatenated log.
        """
        return pd.concat([v.get_log() for v in self.vehicles])

    def __repr__(self):
        return self.vehicles.__repr__()

    # utils for obscuration code
    # see https://github.com/eugenevinitsky/bayesian_reasoning_traffic/blob/nliu/training/flow/core/kernel/util.py for full doc
    def get_blocked_segments(self, position, target_position, target_orientation, target_length, target_width):
        """Define a line segment that blocks the observation vehicle's line of sight"""
        corner_angle = math.degrees(math.atan(target_width / target_length))
        corner_dist = self.euclidian_distance(target_length / 2, target_width / 2)

        corners = self.get_corners(target_position[0], target_position[1], target_orientation, \
                corner_angle, corner_dist)

        angles = []
        for i, c in enumerate(corners):
            angles.append((i, self.get_angle(position[0] - c[0], position[1] - c[1])))

        max_angle = corners[max(angles, key=lambda x: x[1])[0]]
        min_angle = corners[min(angles, key=lambda x: x[1])[0]]

        return(max_angle, min_angle)

    def get_corners(self, x, y, orientation, corner_angle, corner_dist, center_offset=0):
        corners = []

        adjusted_x = x - center_offset * math.cos(math.radians(orientation))
        adjusted_y = y - center_offset * math.sin(math.radians(orientation))

        t_angle = math.radians((orientation + corner_angle) % 360)
        corners.append((adjusted_x + math.cos(t_angle) * corner_dist, \
                adjusted_y + math.sin(t_angle) * corner_dist))

        t_angle = math.radians((orientation + 180 - corner_angle) % 360)
        corners.append((adjusted_x + math.cos(t_angle) * corner_dist, \
                adjusted_y + math.sin(t_angle) * corner_dist))

        t_angle = math.radians((orientation + 180 + corner_angle) % 360)
        corners.append((adjusted_x + math.cos(t_angle) * corner_dist, \
                adjusted_y + math.sin(t_angle) * corner_dist))

        t_angle = math.radians((orientation - corner_angle) % 360)
        corners.append((adjusted_x + math.cos(t_angle) * corner_dist, \
                adjusted_y + math.sin(t_angle) * corner_dist))

        return corners

    def euclidian_distance(self, x, y):
        """Euclidean distance"""
        return math.sqrt(x**2 + y**2)

    def get_angle(self, x, y):
        """Get angle based on the unit circle"""
        if x == 0:
            if y > 0:
                return 90
            else:
                return 270
        elif x < 0:
            return math.degrees(math.atan(y / x)) + 180

        return math.degrees(math.atan(y / x))

    def observed(self, position, orientation, target_position, fov=180, looking_distance=50):
        """Check if a single vehicle/pedestrian can see another vehicle/pedestrian (no obscuration check here)"""
        delta_x = target_position[0] - position[0]
        delta_y = target_position[1] - position[1]

        # edge case where both objects are at the same position
        if delta_x == 0 and delta_y == 0:
            return True

        # object is too far
        if self.euclidian_distance(delta_x, delta_y) > looking_distance:
            return False

        angle = self.get_angle(delta_x, delta_y)
        right_angle = (orientation - angle) % 360
        left_angle = (angle - orientation) % 360

        # object is not in FOV
        if left_angle > fov/2.0 and right_angle > fov/2.0:
            return False
        
        return True

    def check_blocked(self, position, target_position, blocked, vehicle_id):
        """Check if a target vehicle is blocked by another vehicle or object."""
        for b in list(blocked):
            if b == vehicle_id:
                continue
            line_of_sight = (position, target_position)
            if self.lines_intersect(line_of_sight, blocked[b]):
                return True
        return False

    def lines_intersect(self, line1, line2):
        """Check if two lines intersect."""
        def ccw(A, B, C):
                return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])

        a, b, c, d = line1[0], line1[1], line2[0], line2[1]
        return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)