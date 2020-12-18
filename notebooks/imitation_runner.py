import numpy as np
import gym
import highway_env

from highway_env.vehicle.l012vehicles import L0Vehicle, L1Vehicle, L2Vehicle, Pedestrian, FullStop
from notebooks.notebook_utils.cli_parser import create_parser

def run(args):

    env_config = dict()
    if args.scenario == 1:
        env_config["scenario"] = args.scenario
        env_config["spawn_probability"] = 0    # to avoid constant inflow
        env_config["inference_noise_std"] = args.inference_noise     # inference noise (std of normal noise added to inferred accelerations)
        env_config["other_vehicles_type"] = 'highway_env.vehicle.behavior.IDMVehicle' # default behavior for cars is IDM

    env = gym.make('intersection-pedestrian-v0', config=env_config)
    done = False

    obs = env.reset()
    while not done:
        # agent = agent_factory(config)
        # action = agent.act(obs)
        action = None # if action is None, do normal controls
        obs, reward, done, info = env.step(action)
        env.render()

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    run(args)