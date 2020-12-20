import numpy as np
import gym
import highway_env

from highway_env.vehicle.l012vehicles import L0Vehicle, L1Vehicle, L2Vehicle, Pedestrian, FullStop
from notebooks.notebook_utils.cli_parser import create_parser

def run(args):

    env_config = dict()
    if args.scenario == 1:
        env_config["scenario"] = args.scenario
        env_config["spawn_probability"] = 0    # to avoid constant inflow
        env_config["inference_noise_std"] = args.inference_noise     # inference noise (std of normal noise added to inferred accelerations)
        env_config["other_vehicles_type"] = 'highway_env.vehicle.behavior.IDMVehicle' # default behavior for cars is IDM

    env = gym.make('intersection-pedestrian-v0', config=env_config)
    done = False

    # print('Loading policy from...', args.params['learned_policy_file'])
    # import ipdb; ipdb.set_trace()
    # self.loaded_expert_policy = MLPPolicySL(self.rl_trainer.agent.agent_params['ac_dim'],
    #                                         self.rl_trainer.agent.agent_params['ob_dim'],
    #                                         self.rl_trainer.agent.agent_params['n_layers'],
    #                                         self.rl_trainer.agent.agent_params['size'],
    #                                         discrete=self.rl_trainer.agent.agent_params['discrete'],
    #                                         learning_rate=self.rl_trainer.agent.agent_params['learning_rate'])
    # self.loaded_expert_policy.load_state_dict(torch.load(self.params['learned_policy_file'])) 
    # obs = torch.zeros(1, 1, 28).to(torch.device("cuda"))
    # self.loaded_expert_policy.mean_net(obs)                                       
    # print('Done restoring learned policy...')
    # self.collect_policy=self.loaded_expert_policy

    obs = env.reset()
    while not done:
        # agent = agent_factory(config)
        # action = agent.act(obs)
        action = None # if action is None, do normal controls
        obs, reward, done, info = env.step(action)
        env.render()

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    run(args)