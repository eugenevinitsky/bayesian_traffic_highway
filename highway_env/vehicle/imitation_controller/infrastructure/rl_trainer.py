from collections import OrderedDict
import time
import pickle
import numpy as np

import gym
import torch

from highway_env.vehicle.imitation_controller.infrastructure import pytorch_util as ptu
from highway_env.vehicle.imitation_controller.infrastructure.logger import Logger
from highway_env.vehicle.imitation_controller.infrastructure import utils

# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40  # we overwrite this in the code below

class RL_Trainer:

    def __init__(self, params):

        # Get params, create logger, create TF session
        self.params = params
        self.logger = Logger(self.params['logdir'])

        # Set random seeds
        seed = self.params['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        ptu.init_gpu(
            use_gpu=not self.params['no_gpu'],
            gpu_id=self.params['which_gpu']
        )

        # Env
        self.env = gym.make(self.params['env_name'], config=self.params['env_config'])
        self.env.seed(seed)

        # Maximum length for episodes
        self.params['ep_len'] = self.params['ep_len'] or self.env.spec.max_episode_steps
        MAX_VIDEO_LEN = self.params['ep_len']

        # Is this env continuous, or self.discrete?
        discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        self.params['agent_params']['discrete'] = discrete

        # Observation and action sizes
        ob_dim = self.env.observation_space.shape[0]
        ac_dim = self.env.action_space.n if discrete else self.env.action_space.shape[0]
        self.params['agent_params']['ac_dim'] = ac_dim
        self.params['agent_params']['ob_dim'] = ob_dim

        # # simulation timestep, will be used for video saving
        # if 'model' in dir(self.env):
        #     self.fps = 1/self.env.model.opt.timestep
        # else:
        #     self.fps = self.env.env.metadata['video.frames_per_second']

        # agent
        agent_class = self.params['agent_class']
        self.agent = agent_class(self.env, self.params['agent_params'])

    def run_training_loop(self, n_iter, collect_policy, relabel_with_expert, start_relabel_with_expert, expert_policy=None, collect_expert_data=False):
        """
        @Params
        n_iter: number of iterations
        collect_policy: policy used to step through the env and take actions
        relabel_with_expert: whether or not to perform dagger       
        start_relabel_with_expert: iteration at which to start relabel with expert
        collect_expert_data: whether or not to simply run the env using the expert and collect the data of the expert
        expert_policy: expert policy NB this may be a string 
            e.g. "rule_based" where we'd use an 'L0' controller
            yes, this is breaking abstraction barriers - might fix it later 
        """
        # init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()

        for itr in range(n_iter):
            print("\n\n********** Iteration %i ************"%itr)

            training_returns = self.collect_training_trajectories(itr, collect_policy, self.params['batch_size'], expert_policy, collect_expert_data)
            paths, envsteps_this_batch, train_video_paths = training_returns
            self.total_envsteps += envsteps_this_batch

            if relabel_with_expert and itr >= start_relabel_with_expert:
                paths = self.do_relabel_with_expert(expert_policy, paths)
            
            self.agent.add_to_replay_buffer(paths)
            training_logs = self.train_agent()

        if self.params['save_params']:
            print('\nSaving agent params')
            self.agent.save('{}.pt'.format(self.params['logdir']))

    def collect_training_trajectories(self, itr, collect_policy, batch_size, expert_policy=None, collect_expert_data=False):
        """
        @Params
        itr: iteration number
        collect_policy: policy used to take actions and train
        batch_size: number of transitions to collect
        collect_expert_data: whether or not we should use the expert policy to collect data

        @Return
        paths: a list trajectories
        envsteps_this_batch: the sum over the numbers of environment steps in paths
        train_video_paths: paths which also contain videos for visualization purposes
        """

        print("\nCollecting data to be used for training...")
        paths, envsteps_this_batch = utils.sample_trajectories(self.env, collect_policy, batch_size, self.params['ep_len'], expert_policy, collect_expert_data)
        train_video_paths = None
        return paths, envsteps_this_batch, train_video_paths

    def train_agent(self):
        print('\nTraining agent using sampled data from replay buffer...')
        all_logs = []
        for train_step in range(self.params['num_agent_train_steps_per_iter']):
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.agent.sample(self.params['train_batch_size'])
            train_log = self.agent.train(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)
            all_logs.append(train_log)

        return all_logs
