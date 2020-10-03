import argparse
import numpy as np
import gym
import highway_env

from matplotlib import pyplot as plt
from multiprocessing import Pool


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

    # REMOVE THIS
    # env.vehicle == ego vehicle
    # env.road.vehicles == list of all vehicles on the road

    env.reset()
    for _ in range(50):
        action = [.334] # accel between -1 and 1
        obs, reward, done, info = env.step(action)
        if reward < -4:
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
        
