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

and lines 292 - 340, with first few lines:

```
    def scenario_1(train=False, noise=0):
        prob_collect_non_ego_data = np.random.uniform(0, 1) if train else 0
        spawn_vehicle(scenario=1, vclass=L0Vehicle, lane=("o1", "ir1", 0), dest="o3", pos=70 + noise * 5, speed=8.0 + noise * 2, type="car", controlled=prob_collect_non_ego_data < 0.5) # controlled car for the actual scenario
        spawn_vehicle(scenario=1, vclass=L1Vehicle if not train else L0Vehicle, lane=("o3", "ir3", 0), dest="o1", pos=75 + noise * 5, speed=14 + noise * 2, type="car", controlled=prob_collect_non_ego_data >= 0.5)
        if train and np.random.uniform(0, 1) > 0.5:
            spawn_vehicle(scenario=1, vclass=Pedestrian, lane=("p2", "p2_end", 0), dest="p2_end", pos=2, speed=2.0, type="ped")
        spawn_vehicle(scenario=1, vclass=FullStop, lane=("o2", "ir2", 0), dest="o1", pos=97, speed=0.0, type="bus")
```

The stepping of the env occurs in `highway_env/vehicle/imitation_controller/infrastructure/utils.py`

To use a trained (DAgger) policy, add the flag

`--collect_policy_path <path_to_trained_policy>` (excuse the poor naming) and specify the scenario to use (i.e something that's not scenario 11)

@fixme Small bug with `get_ped_state` in `l012vehicles.py` 