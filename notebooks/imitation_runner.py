import numpy as np
import gym
import torch
import highway_env

from highway_env.vehicle.l012vehicles import L0Vehicle, L1Vehicle, L2Vehicle, Pedestrian, FullStop
from notebooks.notebook_utils.cli_parser import create_parser, get_cli_params
from highway_env.vehicle.imitation_controller.policies.MLP_policy import MLPPolicySL

import cProfile
import pandas as pd

def run(args, params):

    env_config = dict()
    if args.scenario == 1:
        env_config["scenario"] = args.scenario
        env_config["spawn_probability"] = 0    # to avoid constant inflow
        env_config["inference_noise_std"] = args.inference_noise     # inference noise (std of normal noise added to inferred accelerations)
        env_config["other_vehicles_type"] = 'highway_env.vehicle.behavior.IDMVehicle' # default behavior for cars is IDM

    env = gym.make('intersection-pedestrian-v0', config=env_config)
    done = False
    agent_params = {
        'n_layers': params['n_layers'],
        'size': params['size'],
        'learning_rate': params['learning_rate'],
        'max_replay_buffer_size': params['max_replay_buffer_size'],
        'discrete': False
        }  
    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]
    agent_params['ac_dim'] = ac_dim
    agent_params['ob_dim'] = ob_dim

    # actor/policy
    collect_policy = MLPPolicySL(
        agent_params['ac_dim'],
        agent_params['ob_dim'],
        agent_params['n_layers'],
        agent_params['size'],
        discrete=agent_params['discrete'],
        learning_rate=agent_params['learning_rate'],
    )
    
    collect_policy.load_state_dict(torch.load(params['collect_policy_path']))

    obs = env.reset()
    env.vehicle.trained_policy = collect_policy
    # let us do the state normaliziation in l0l1l2 by setting the attributes from self.env.observation_type.features_range
    while not done:
        action = collect_policy.get_action(obs)[0]
        obs, reward, done, info = env.step(action)
        env.render()

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    params = get_cli_params()
    run(args, params)
