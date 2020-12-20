import os
import time
import torch

from highway_env.vehicle.imitation_controller.scripts.imitation_cli_parser import get_cli_params
from highway_env.vehicle.imitation_controller.agents.imitation_agent import ImitationAgent
from highway_env.vehicle.imitation_controller.infrastructure.rl_trainer import RL_Trainer
from highway_env.vehicle.imitation_controller.policies.MLP_policy import MLPPolicySL

class ImitationTrainer:
    def __init__(self, params):

        agent_params = {
            'n_layers': params['n_layers'],
            'size': params['size'],
            'learning_rate': params['learning_rate'],
            'max_replay_buffer_size': params['max_replay_buffer_size'],
            }

        self.params = params
        self.params['agent_class'] = ImitationAgent 
        self.params['agent_params'] = agent_params

        # RL TRAINER
        self.rl_trainer = RL_Trainer(self.params) 

        # initalize collect_policy
        if self.params['collect_policy_path']:
            self.collect_policy = MLPPolicySL(self.rl_trainer.agent.agent_params['ac_dim'],
                                                    self.rl_trainer.agent.agent_params['ob_dim'],
                                                    self.rl_trainer.agent.agent_params['n_layers'],
                                                    self.rl_trainer.agent.agent_params['size'],
                                                    discrete=self.rl_trainer.agent.agent_params['discrete'],
                                                    learning_rate=self.rl_trainer.agent.agent_params['learning_rate'])
            self.collect_policy.load_state_dict(torch.load(self.params['collect_policy_path']))
            self.rl_trainer.agent.actor = self.collect_policy 
            print('Done restoring learned collect policy from', self.params['collect_policy_path'])

        self.collect_policy = self.rl_trainer.agent.actor   

        # initalize expert_policy
        if isinstance(self.params['expert_policy_path'], str):
            self.expert_policy = self.params['expert_policy_path']
        else:
            print(f'expert policy is a policy; not implemented yet')
            assert False

    def run_training_loop(self):
        if self.params['expert_policy_path'] == 'rule_based':
            self.expert_policy = 'rule_based'

        self.rl_trainer.run_training_loop(
            n_iter=self.params['n_iter'],
            # initial_expertdata=self.params['expert_data'],
            collect_policy=self.collect_policy,
            # eval_policy=self.rl_trainer.agent.actor,
            relabel_with_expert=self.params['do_dagger'],
            expert_policy=self.expert_policy,
            start_relabel_with_expert=False # for now
        )

if __name__ == '__main__':
    params = get_cli_params()
    ## directory for logging
    imitation_trainer = ImitationTrainer(params)
    imitation_trainer.run_training_loop()