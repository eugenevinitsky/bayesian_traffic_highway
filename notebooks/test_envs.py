import argparse
import numpy as np
import gym
import highway_env
import copy
import random

from matplotlib import pyplot as plt
from multiprocessing import Pool


"""

Scenarios 1 and 9 (L0 & L1 only) and 10 (w/ L2) are operational 
    (functional but design can be improving, ie vehicles leaving once ped is gone (cf if True in l012vehicles.py))


"""

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

        # initialize values
        done = False
        first_action = None
        returns = 0.0

        # compute sequence of actions to be done
        binary = (("0" * 40) + "{0:b}".format(action_sequence))[::-1]

        for i in range(num_steps_per_simulation):
            # get the two possible actions (decel or accel)
            action_decel = -1
            action_accel = env_clone.vehicle.acceleration(env_clone.vehicle)

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


def run(scenario=1, inference_noise_std=0.0):
    env = gym.make('intersection-pedestrian-v0')

    # pick custom scenario
    env.config["scenario"] = scenario

    # to avoid constant inflow
    env.config["spawn_probability"] = 0

    # inference noise (std of normal noise added to inferred accelerations)
    env.config["inference_noise_std"] = inference_noise_std

    # default behavior for cars is IDM
    env.config["other_vehicles_type"] = 'highway_env.vehicle.behavior.IDMVehicle'

    env.reset()
    for _ in range(50):
        if scenario in [10]:
            action = [MPC(env, num_steps_per_simulation=5)]
        else:
            action = [0] # accel between -1 and 1 (doesnt have an impact)
        obs, reward, done, info = env.step(action)

        reward = compute_reward(env)
        if reward < 0:
            print('CRASH')
            return False

        env.render()

    # plot 1
    if False:
        plot_data = env.controlled_vehicles[0].plot_data
        l0_accel = np.array(plot_data['l0_accel'])
        l1_accel = np.array(plot_data['l1_accel'])
        ped_probs = np.array(plot_data['ped_probs'])

        print(l0_accel.shape, l1_accel.shape, ped_probs.shape)
        
        plt.figure()
        plt.plot(l0_accel, label='L0 accel')
        plt.plot(l1_accel, label='L1 accel')
        for i in range(4):
            plt.plot(ped_probs[:,i], label=f"Ped probs ({['south', 'west', 'north', 'east'][i]})", linestyle='--')
        plt.legend(loc='lower center')
        plt.title('Scenario 1')
        plt.show()

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario",
                        help="specify experiment number - 0, 1, 2, 3, 9 or 10",
                        type=int,
                        default=1)

    args = parser.parse_args()
    scenario = args.scenario
    run(scenario)

    # plots 2
    if False:
        def f(noise):
            np.random.seed(int(noise*10000))
            print(noise)
            return run(scenario=1, inference_noise_std=noise)

        with Pool(17) as p:
            lst = [50 + i/1000 for i in range(50)]
            res = p.map(f, lst)
            print(res)
            print(np.count_nonzero(res) / len(res))


    if False:
        def f(noise, scenario=1, n_runs=50):
            results = []
            for i in range(n_runs):
                print(i)
                results.append(run(scenario=scenario, inference_noise_std=noise))
            return results

        noises = list(range(0, 101, 5))
        y = []
        with Pool(len(noises)) as p:
            res = p.map(f, noises)
            for n, r in zip(noises, res):
                r = np.array(r)
                true = np.count_nonzero(r)
                false = len(r) - true
                prct = true / (true + false)
                y.append(prct)

        plt.figure()
        plt.title('Scenario 1')
        plt.xlabel('Inference std noise')
        plt.ylabel('% of scenarios that dont crash (over 50 scenarios)')
        plt.plot(noises, y)
        plt.show()
        
