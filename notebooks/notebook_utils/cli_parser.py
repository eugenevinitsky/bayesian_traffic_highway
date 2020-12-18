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
    
    return parser

