import gym
import highway_env

for env_class in [
    highway_env.envs.RoundaboutEnv,
    highway_env.envs.IntersectionEnv,
]:
  env = env_class(config=dict(action=dict(type="L0Action")))
  done = False
  env.reset()

  while not done:
    obs, reward, done, info = env.step(None)
    env.render()