import copy
import random

import gym
import highway_env

env = gym.make("intersection-v0")
obs = env.reset()
done = False


def MPC(environment, num_simulations, num_steps_per_simulation):
  action_returns = list()
  for _ in range(num_simulations):
    env_clone = copy.deepcopy(environment)
    done = False
    first_action = None
    returns = 0.0
    for _ in range(num_steps_per_simulation):
      action = random.randint(0, env_clone.action_space.n - 1)
      _, reward, done, _ = env_clone.step(action)
      returns += reward
      if first_action is None:
        first_action = action
      if done:
        break
    action_returns.append((first_action, returns))
  ascending_action_returns = sorted(action_returns, key=lambda x: x[1])
  best_action = ascending_action_returns[-1][0]
  # import pdb
  # pdb.set_trace()
  return best_action


while not done:
  action = MPC(env, num_simulations=10, num_steps_per_simulation=5)
  obs, reward, done, info = env.step(action)
  env.render()