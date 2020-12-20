import argparse

def create_parser():
    """Create the parser to capture CLI arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='[TOM] Runs a traffic scenario for bayesian reasoning')

    # required input parameters
    parser.add_argument('--scenario', type=int, required=True, help='Specific the scenario to run. Options: 1 (4 way intersection)')
    # optional input parameters
    parser.add_argument(
        '--num_rollouts',
        type=int,
        default=1,
        help='The number of rollouts to visualize.')
    parser.add_argument(
        '--render',
        action='store_true',
        help='Decide if we want to render the scenario')
    parser.add_argument(
        '--horizon',
        type=int,
        default=30,
        help='Specifies the horizon.')
    # optional input parameters
    parser.add_argument(
        '--inference_noise',
        type=int,
        default=0,
        help='Inference noise')
    

    # learned policy parameters
    parser.add_argument('--expert_policy_path', type=str, default='', help='Specify the expert policy path. Options: "rule_based", "human_input" #TODO (not supported yet), an actual neural net policy path')
    parser.add_argument('--collect_policy_path', type=str, default='', help="Collect policy is the 'beginner' neural net policy that gets trained and steps the environment. If empty string, create a new neural net policy")
    parser.add_argument('--max_replay_buffer_size', type=int, default=1000000)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', type=int, default=0)

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

    return parser


def get_cli_params():
    """
    Return cli params as a dict
    """
    parser = create_parser()
    args = parser.parse_args()
    params = vars(args)

    env_config = dict()
    if params['scenario'] == 1:
        env_config["scenario"] = args.scenario
        env_config["spawn_probability"] = 0    # to avoid constant inflow
        env_config["inference_noise_std"] = 0     # inference noise (std of normal noise added to inferred accelerations)
        env_config["other_vehicles_type"] = 'highway_env.vehicle.behavior.IDMVehicle' # default behavior for cars is IDM
        params['env_config'] = env_config
    
    params['logdir'] = ''

    return params
