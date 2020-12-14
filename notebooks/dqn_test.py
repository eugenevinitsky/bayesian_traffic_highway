# Environment
# !pip install git+https://github.com/eleurent/highway-env#egg=highway-env
import gym
import highway_env

# Agent
# !pip install git+https://github.com/eleurent/rl-agents#egg=rl-agents

# Visualisation utils
import sys
# %load_ext tensorboard
# !pip install tensorboardx gym pyvirtualdisplay
# !apt-get install -y xvfb python-opengl ffmpeg
# !git clone https://github.com/eleurent/highway-env.git
# sys.path.insert(0, '/content/highway-env/scripts/')
# from utils import show_videos

from rl_agents.trainer.evaluation import Evaluation
from rl_agents.agents.common.factory import load_agent, load_environment

# Get the environment and agent configurations from the rl-agents repository
# !git clone https://github.com/eleurent/rl-agents.git
# %cd /content/rl-agents/scripts/


env_config = '/home/thankyou-always/TODO/research/bayesian_traffic_highway/scripts/rl-agents/scripts/configs/IntersectionEnv/env.json'
agent_config = '/home/thankyou-always/TODO/research/bayesian_traffic_highway/scripts/rl-agents/scripts/configs/IntersectionEnv/agents/DQNAgent/ego_attention_2h.json'

env = load_environment(env_config)
agent = load_agent(agent_config, env)
evaluation = Evaluation(env, agent, num_episodes=3000, display_env=False)
print(f"Ready to train {agent} on {env}")

evaluation.train()

env = load_environment(env_config)
env.configure({"offscreen_rendering": True})
agent = load_agent(agent_config, env)
evaluation = Evaluation(env, agent, num_episodes=3, recover=True)
evaluation.test()
show_videos(evaluation.run_directory)