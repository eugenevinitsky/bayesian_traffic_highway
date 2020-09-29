from typing import Tuple, Union

import numpy as np

from highway_env.road.road import Road, Route, LaneIndex
from highway_env.types import Vector
from highway_env.vehicle.controller import ControlledVehicle
from highway_env import utils
from highway_env.vehicle.kinematics import Vehicle
from highway_env.road.objects import RoadObject


class FullStop(ControlledVehicle):
    def acceleration(self,
                     ego_vehicle: ControlledVehicle,
                     front_vehicle: Vehicle = None,
                     rear_vehicle: Vehicle = None) -> float:        
        return -40

class L0Vehicle(ControlledVehicle):
    """Rule based vehicle that obeys normal traffic rules"""
    
    def acceleration(self,
                     ego_vehicle: ControlledVehicle,
                     front_vehicle: Vehicle = None,
                     rear_vehicle: Vehicle = None) -> float:
        raise NotImplementedError

class L1Vehicle(L0Vehicle):
    """Rule based vehicle (L0) operating on an inferred belief state"""
    
    def acceleration(self,
                     ego_vehicle: ControlledVehicle,
                     front_vehicle: Vehicle = None,
                     rear_vehicle: Vehicle = None) -> float:
        raise NotImplementedError

class L2Vehicle(L0Vehicle):
    """L2 vehicle"""
    
    def acceleration(self,
                     ego_vehicle: ControlledVehicle,
                     front_vehicle: Vehicle = None,
                     rear_vehicle: Vehicle = None) -> float:
        raise NotImplementedError