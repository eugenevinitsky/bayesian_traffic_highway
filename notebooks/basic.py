import gym
import highway_env

def run(scenario=10):
    env = gym.make('intersection-pedestrian-v0')

    model = 1

    obs = env.reset()
    env.render()
    env.
    for _ in range(1000):
        
        action = env.action_space.sample()
        obs, reward, done, info = env.step(1)

        env.render()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario",
                        help="specify experiment number - 0, 1, 2, 3, 9 or 10",
                        type=int,
                        default=1)

    args = parser.parse_args()
    scenario = args.scenario
    run(scenario, inference_noise_std=0)