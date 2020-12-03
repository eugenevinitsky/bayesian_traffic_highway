# highway-env

Original code: https://github.com/eleurent/highway-env

A collection of environments for *autonomous driving* and tactical decision-making tasks

## Installation

`pip install --user git+https://github.com/eleurent/highway-env`

## Usage 

- Run with `python notebooks/test_envs.py --scenario <1|2|9|10>`
- Inference for L1 cars can be enabled/disabled in `vehicle/l012vehicles.py` line 106
- L2 vehicles can be disabled by replacing `L2Vehicle` by `L0Vehicle` in `envs/intersection_with_pedestrian_env.py` (line 262 for scenario 2, 275 for scenario 10)