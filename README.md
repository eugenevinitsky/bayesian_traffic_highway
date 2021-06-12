# highway-env

Original code: https://github.com/eleurent/highway-env

A collection of environments for *autonomous driving* and tactical decision-making tasks

## Installation

`pip install --user git+https://github.com/eleurent/highway-env`

## Usage 

- Run with `python notebooks/test_envs.py --scenario <1|2|9|10>`
- Inference for L1 cars can be enabled/disabled in `vehicle/l012vehicles.py` line 106
- L2 vehicles can be disabled by replacing `L2Vehicle` by `L0Vehicle` in `envs/intersection_with_pedestrian_env.py` (line 262 for scenario 2, 275 for scenario 10)

### DAgger

To collect data on all the scenarios, run: `python vehicle/imitation_controller/scripts/train_imitation_agent.py --env_name intersection-pedestrian-v0 --exp_name <some_exp_name> --save_params <True/False> --scenario 11 --expert_policy rule_based --render` 

See `highway_env/vehicle/imitation_controller/scripts/imitation_cli_parser.py` for details on parameters

We randomize the env in `intersection_with_pedestrian_env.py` lines 104 - 109:
```
    self.config["true_scenario"] = self.config["scenario"]
    if self.config["scenario"] == 11:
        self.config["true_scenario"] = np.random.choice([1, 2, 9, 10])
        self.config["train"] = True
        self.config["train_noise"] = np.random.uniform()
        print(f'scenario: {self.config["scenario"]}')
```

The stepping of the env occurs in `highway_env/vehicle/imitation_controller/infrastructure/utils.py`
