import os
import time

from highway_env.vehicle.imitation_controller.scripts.imitation_cli_parser import get_cli_params
from highway_env.vehicle.imitation_controller.agents.imitation_agent import ImitationAgent
from highway_env.vehicle.imitation_controller.infrastructure.rl_trainer import RL_Trainer

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

    def run_training_loop(self):
        if self.params['expert_policy_path'] and self.params['expert_policy_path'] == 'rule_based':
            self.expert_policy = 'rule_based'
        self.rl_trainer.run_training_loop(
            n_iter=self.params['n_iter'],
            # initial_expertdata=self.params['expert_data'],
            collect_policy=self.rl_trainer.agent.actor,
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