import gym
import highway_env
from matplotlib import pyplot as plt

env = gym.make('intersection-pedestrian-v0')
env.config["lanes_count"] = 2
env.reset()
for _ in range(20):
    action = env.action_type.actions_indexes["IDLE"]
    obs, reward, done, info = env.step(action)
    env.render()
    import ipdb; ipdb.set_trace()

# plt.imshow(env.render(mode="rgb_array"))
# plt.show()
