import numpy as np
import gym
import torch
import highway_env

from highway_env.vehicle.l012vehicles import L0Vehicle, L1Vehicle, L2Vehicle, Pedestrian, FullStop
from notebooks.notebook_utils.cli_parser import create_parser, get_cli_params
from highway_env.vehicle.imitation_controller.policies.MLP_policy import MLPPolicySL

import cProfile
import pandas as pd

# MPC stuff
import copy
import random
from multiprocessing import Pool

def compute_reward(env):
    """Compute the reward ourselves
    Rwd = speed of ego vehicle - 100 * number of crashed vehicles"""
    reward = env.vehicle.speed
    for v in env.road.vehicles:
        if v.crashed:
            reward -= 100
    return reward

def MPC(environment, num_steps_per_simulation):
    """Performs MPC for L2 actions"""
    # to keep track of the return of each action for each simulation
    action_returns = list()

    # simulate 2^n simulations
    for action_sequence in range(2 ** num_steps_per_simulation):
        # copy environment so there's no need to reset anything
        env_clone = copy.deepcopy(environment)
        # reset L1 priors since L2 can't know them
        env_clone.reset_priors()

        # remove vehicles that can't be seen initially so we don't simulate them
        visible_vehs = env_clone.road.close_vehicles_to(env_clone.vehicle, obscuration=True)
        for v in env_clone.road.vehicles:
            if v not in visible_vehs and v != env_clone.vehicle and isinstance(v, Pedestrian):
                env_clone.road.vehicles.remove(v)

        # initialize values
        done = False
        first_action = None
        returns = 0.0

        # compute sequence of actions to be done
        binary = (("0" * 40) + "{0:b}".format(action_sequence))[::-1]

        for i in range(num_steps_per_simulation):
            # get the two possible actions (decel or accel)
            action_decel = -1
            action_accel = 1 #env_clone.vehicle.accel_no_ped() #env_clone.vehicle)

            # get current action for current sequence
            action = [action_decel, action_accel][int(binary[i])]
            # execute action in copied environment
            env_clone.step([action])
            # compute reward
            returns += compute_reward(env_clone)

            if first_action is None:
                first_action = action
            if done:
                break

        action_returns.append((first_action, returns))

    # get first action that yielded maximum return
    ascending_action_returns = sorted(action_returns, key=lambda x: x[1])
    best_action = ascending_action_returns[-1][0]

    return best_action

def run(args, params):

    env_config = dict()
    if args.scenario == 1 or args.scenario == 10:
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
        # action = collect_policy.get_action(obs)[0]
        obs, reward, done, info = env.step(None)
        env.render()




if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    params = get_cli_params()
    run(args, params)
