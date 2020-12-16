import os
import time
import pickle
import torch

from highway_env.vehicle.imitation_controller.infrastructure.rl_trainer import RL_Trainer
from highway_env.vehicle.imitation_controller.agents.bc_agent import BCAgent
from highway_env.vehicle.imitation_controller.policies.loaded_gaussian_policy import LoadedGaussianPolicy
from highway_env.vehicle.imitation_controller.policies.MLP_policy import MLPPolicySL

class BC_Trainer(object):

    def __init__(self, params):

        #######################
        ## AGENT PARAMS
        #######################

        agent_params = {
            'n_layers': params['n_layers'],
            'size': params['size'],
            'learning_rate': params['learning_rate'],
            'max_replay_buffer_size': params['max_replay_buffer_size'],
            }

        self.params = params
        self.params['agent_class'] = BCAgent ## HW1: you will modify this
        self.params['agent_params'] = agent_params

        ################
        ## RL TRAINER
        ################

        self.rl_trainer = RL_Trainer(self.params) ## HW1: you will modify this

        #######################
        ## LOAD EXPERT POLICY
        #######################

        print('Loading policy from...', self.params['learned_policy_file'])
        import ipdb; ipdb.set_trace()
        self.loaded_expert_policy = MLPPolicySL(self.rl_trainer.agent.agent_params['ac_dim'],
                                                self.rl_trainer.agent.agent_params['ob_dim'],
                                                self.rl_trainer.agent.agent_params['n_layers'],
                                                self.rl_trainer.agent.agent_params['size'],
                                                discrete=self.rl_trainer.agent.agent_params['discrete'],
                                                learning_rate=self.rl_trainer.agent.agent_params['learning_rate'])
        self.loaded_expert_policy.load_state_dict(torch.load(self.params['learned_policy_file'])) 
        obs = torch.zeros(1, 1, 28).to(torch.device("cuda"))
        self.loaded_expert_policy.mean_net(obs)                                       
        print('Done restoring learned policy...')

    def run_training_loop(self):

        self.rl_trainer.run_training_loop(
            n_iter=self.params['n_iter'],
            initial_expertdata=None,
            collect_policy=self.rl_trainer.agent.actor,
            eval_policy=self.rl_trainer.agent.actor,
            relabel_with_expert=self.params['do_dagger'],
        )

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--expert_policy_file', '-epf', type=str, default='/home/thankyou-always/TODO/research/bayesian_traffic_highway/highway_env/vehicle/data/q1_1_intersection-pedestrian-v0_15-12-2020_23-47-18/policy_itr_0.pt', required=False)  # relative to where you're running this script from
    parser.add_argument('--learned_policy_file', '-lpf', type=str, default='/home/thankyou-always/TODO/research/bayesian_traffic_highway/highway_env/vehicle/data/q1_1_intersection-pedestrian-v0_15-12-2020_23-47-18/policy_itr_0.pt', required=False)  # relative to where you're running this script from
    parser.add_argument('--expert_data', '-ed', type=str, required=False) #relative to where you're running this script from
    parser.add_argument('--env_name', '-env', type=str, help='choices: Ant-v2, Humanoid-v2, Walker-v2, HalfCheetah-v2, Hopper-v2', required=True)
    parser.add_argument('--exp_name', '-exp', type=str, default='pick an experiment name', required=True)
    parser.add_argument('--do_dagger', action='store_true')
    parser.add_argument('--ep_len', type=int, default=1)

    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1)  # number of gradient steps for training policy (per iter in n_iter)
    parser.add_argument('--n_iter', '-n', type=int, default=1)

    parser.add_argument('--batch_size', type=int, default=1)  # training data collected (in the env) during each iteration
    parser.add_argument('--eval_batch_size', type=int,
                        default=1)  # eval data collected (in the env) for logging metrics
    parser.add_argument('--train_batch_size', type=int,
                        default=1)  # number of sampled data points to be used per gradient/train step

    parser.add_argument('--n_layers', type=int, default=2)  # depth, of policy to be learned
    parser.add_argument('--size', type=int, default=64)  # width of each layer, of policy to be learned
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)  # LR for supervised learning

    parser.add_argument('--video_log_freq', type=int, default=5)
    parser.add_argument('--scalar_log_freq', type=int, default=1)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', type=int, default=0)
    parser.add_argument('--max_replay_buffer_size', type=int, default=1000000)
    parser.add_argument('--save_params', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--tune_hyperparam', type=bool, default=False)
    parser.add_argument('--inference_noise_std', type=int, default=0)
    parser.add_argument("--scenario",
                    help="specify experiment number - 0, 1, 2, 3, 9 or 10",
                    type=int,
                    default=1)
    args = parser.parse_args()

    # convert args to dictionary
    params = vars(args)

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    if args.do_dagger:
        # Use this prefix when submitting. The auto-grader uses this prefix.
        logdir_prefix = 'q2_'
        assert args.n_iter>1, ('DAGGER needs more than 1 iteration (n_iter>1) of training, to iteratively query the expert and train (after 1st warmstarting from behavior cloning).')
    else:
        # Use this prefix when submitting. The auto-grader uses this prefix.
        logdir_prefix = 'q1_'
        assert args.n_iter==1, ('Vanilla behavior cloning collects expert data just once (n_iter=1)')

    ## directory for logging
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = logdir_prefix + args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)


    ###################
    ### RUN TRAINING
    ###################
    # with open(params['expert_data'], 'rb') as f:
    #     data = pickle.loads(f.read())
    
    if not args.tune_hyperparam:
        trainer = BC_Trainer(params)
        trainer.run_training_loop()
    else:
        # THIS CHUNK OF CODE IS FOR Q1.3 tuning hyper parameter of num_agent_train_steps_per_iter
        num_agent_train_steps_per_iter_lst = [i * 100 for i in range(1, 21)]
        for tr_steps in num_agent_train_steps_per_iter_lst:
            params['num_agent_train_steps_per_iter'] = tr_steps
            trainer = BC_Trainer(params)
            trainer.run_training_loop()

if __name__ == "__main__":
    main()

