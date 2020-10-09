import argparse
import numpy as np
import gym
import highway_env
import copy
import random
from multiprocessing import Pool

from highway_env.vehicle.l012vehicles import L0Vehicle, L1Vehicle, L2Vehicle, Pedestrian, FullStop

# matplotlib config
import matplotlib.pyplot as plt
from matplotlib import cycler
plt.style.use("default")
plt.rcParams.update(
  {"lines.linewidth": 1.0,
   "axes.grid": True,
   "grid.linestyle": ":",
   "axes.grid.axis": "both",
   "axes.prop_cycle": cycler('color',
                             ['0071bc', 'd85218', 'ecb01f',
                              '7d2e8d', '76ab2f', '4cbded', 'a1132e']),
   "xtick.top": True,
   "xtick.minor.size": 0,
   "xtick.direction": "in",
   "xtick.minor.visible": True,
   "ytick.right": True,
   "ytick.minor.size": 0,
   "ytick.direction": "in",
   "ytick.minor.visible": True,
   "legend.framealpha": 1.0,
   "legend.edgecolor": "black",
   "legend.fancybox": False,
   "figure.figsize": (3, 6),
   "figure.autolayout": False,
   "savefig.dpi": 300,
   "savefig.format": "pdf",
   "savefig.bbox": "tight",
   "savefig.pad_inches": 0.01,
   "savefig.transparent": True
  }
)
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


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

    # number of simulation steps
    n_steps = 30

    env.reset()
    for i in range(n_steps):
        if scenario in [2, 10]:
            # scenarios with L2 controller
            if False:
                # hardcode MPC action for fast plotting (this is for scenario 2)
                if i == 0: action = [-1]
                elif i == 1: action = [1]
                elif i == 2: action = [-1]
                elif i == 3: action = [1]
                else: action = [1]
            else:
                action = [MPC(env, num_steps_per_simulation=5)]
        else:
            # scenarios without L2 (actions are not computed here)
            action = [0]
        obs, reward, done, info = env.step(action)

        reward = compute_reward(env)
        if reward < 0:
            print('CRASH')
            return False

        env.render()

    # acceleration and probabilities plots
    if True:
        # retrieve plot data
        for v in env.road.vehicles:
            if hasattr(v, 'plot_data'):
                plot_data = v.plot_data
                break

        
        all_data = []
        for k in plot_data.keys():
            if not isinstance(k, str):
                data = plot_data[k]
                times = [d[0] for d in data]
                speeds = [d[1] for d in data]
                accels = [d[2] for d in data]

                # i = 0
                # while i < len(times) and times[i] < 999:
                #     i += 1
                # times = times[:i]
                # speeds = speeds[:i]
                # accels = accels[:i]

                all_data.append((times, speeds, accels, k))

        if scenario == 9:
            # for scenario 9, only plot one L0 vehicle
            all_data = all_data[:2]

        # plt.figure()
        fig, axs = plt.subplots(2, 1, constrained_layout=True)

        if True:
            # accel plot
            for times, speeds, accels, key in all_data:
                if isinstance(key, L0Vehicle): lb = 'L0'
                if isinstance(key, L1Vehicle): lb = 'L1'
                if isinstance(key, L2Vehicle): lb = 'L2'
                axs[0].plot(times, accels, label=lb)
            axs[0].set_title('Time vs. Acceleration')
            axs[0].set_ylabel('Acceleration (m/s²)')
            axs[0].set_xlabel('Time (s)')
            axs[0].legend()
            
        if True:
            # probabilities plot
            for i in range(4):
                axs[1].plot(plot_data['time'], np.array(plot_data['ped_probs'])[:,i], label=f"{['South', 'West', 'North', 'East'][i]}")
            axs[1].set_title('Time vs. Inferred Probabilities')
            axs[1].set_ylabel('Probability')
            axs[1].set_xlabel('Time (s)')
            axs[1].set_xlim(left=all_data[0][0][0], right=all_data[0][0][-1])
            axs[1].legend()

        plt.savefig(f'figs/scenario_{scenario}.png')
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario",
                        help="specify experiment number - 0, 1, 2, 3, 9 or 10",
                        type=int,
                        default=1)

    args = parser.parse_args()
    scenario = args.scenario
    run(scenario, inference_noise_std=0)

    # noise vs number of crashes plot
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
        plt.title(f'Scenario {scenario}')
        plt.xlabel('Inference std noise')
        plt.ylabel('% of scenarios that dont crash (over 50 scenarios)')
        plt.plot(noises, y)
        plt.show()       
