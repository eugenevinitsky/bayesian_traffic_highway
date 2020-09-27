import argparse
from matplotlib import pyplot as plt

import gym

import highway_env

def run(scenario):
    env = gym.make('intersection-pedestrian-v0')

    # pick custom scenario (implemented: 1, 2, 3, 9)
    env.config["scenario"] = scenario

    # to avoid constant inflow
    env.config["spawn_probability"] = 0
    
    # set controller of non-ego vehicle
    # 'other_vehicles_type': 'highway_env.vehicle.behavior.IDMVehicle'
    if env.config["scenario"] == 10:
        env.config["other_vehicles_type"] = 'highway_env.vehicle.behavior.IDMVehicle'

    env.reset()
    for _ in range(50):
        action = env.action_type.actions_indexes["FASTER"]
        obs, reward, done, info = env.step(action)
        env.render()

# REMOVE THIS
# env.vehicle == ego vehicle
# env.road.vehicles == list of all vehicles on the road

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario",
                        help="specify experiment number - 0, 1, 2, 3, 9 or 10",
                        type=int,
                        default=10)

    args = parser.parse_args()

    scenario = args.scenario
    run(scenario)

