import argparse
import os
import time

def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='[TOM] Runs a traffic scenario to get a blueprint human model using imitation learning + DAgger')

    # required input parameters
    parser.add_argument('--env_name', type=str, default='intersection-pedestrian-v0', required=True, help='Specific the env name')
    parser.add_argument('--exp_name', type=str, default='imitation_learning', required=True, help='Specific the experiment name')
    parser.add_argument('--save_params', type=bool, default=False, required=True)

    parser.add_argument('--scenario', type=int, default=1, required=True, help='Specific the scenario to run. Options: 1 (4 way intersection)')
    # optional input parameters
    parser.add_argument('--expert_policy_path', type=str, default='rule_based', help='Specific the expert policy path name if we wish to load an expert policy, otherwise we can set the expert to run_based')
    parser.add_argument('--collect_expert_data', action='store_true', help='Decide if we want to directly collect the expert trajectories')
    parser.add_argument('--max_replay_buffer_size', type=int, default=1000000)

    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', type=int, default=0)

    parser.add_argument('--render', action='store_true')
    parser.add_argument('--do_dagger', action='store_true')
    parser.add_argument('--ep_len', type=int, default=250)

    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1000)  # number of gradient steps for training policy (per iter in n_iter)
    parser.add_argument('--n_iter', '-n', type=int, default=5)

    parser.add_argument('--batch_size', type=int, default=800)  # training data collected (in the env) during each iteration
    parser.add_argument('--eval_batch_size', type=int,
                        default=5000)  # eval data collected (in the env) for logging metrics
    parser.add_argument('--train_batch_size', type=int,
                        default=100)  # number of sampled data points to be used per gradient/train step

    parser.add_argument('--n_layers', type=int, default=2)  # depth, of policy to be learned
    parser.add_argument('--size', type=int, default=64)  # width of each layer, of policy to be learned
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)  # LR for supervised learning
    
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--tune_hyperparam', type=bool, default=False)
    parser.add_argument('--learned_policy_path', type=str, default='')


    # optional input parameters
    parser.add_argument(
        '--inference_noise',
        type=int,
        default=0,
        help='Inference noise')
    
    return parser


def get_cli_params():
    """
    Return cli params as a dict
    """
    parser = create_parser()
    args = parser.parse_args()
    params = vars(args)

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir


    env_config = dict()
    if params['scenario'] == 1:
        env_config["scenario"] = args.scenario
        env_config["spawn_probability"] = 0    # to avoid constant inflow
        env_config["inference_noise_std"] = 0     # inference noise (std of normal noise added to inferred accelerations)
        env_config["other_vehicles_type"] = 'highway_env.vehicle.behavior.IDMVehicle' # default behavior for cars is IDM
        params['env_config'] = env_config
    

    return params