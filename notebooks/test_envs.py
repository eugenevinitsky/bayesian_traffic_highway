import gym
import highway_env
from matplotlib import pyplot as plt

env = gym.make('intersection-pedestrian-v0')

# pick custom scenario (implemented: 1, 2, 3, 9)
env.config["scenario"] = 3

# to avoid constant inflow
env.config["spawn_probability"] = 0  

env.reset()
for _ in range(50):
    action = env.action_type.actions_indexes["FASTER"]
    obs, reward, done, info = env.step(action)
    env.render()
    # import ipdb; ipdb.set_trace()
