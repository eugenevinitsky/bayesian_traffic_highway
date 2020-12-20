import numpy as np
import time

def sample_trajectory(env, collect_policy, max_path_length, expert_policy, collect_expert_data, render=False):
    """
    Samples a trajectory with maximum steps of max_path_length
    
    Adjusted to assume there are no expert 'policies' in terms of neural nets

    @Param
    env: environment
    collect_policy: neural net policy used to run the env 
    expert_policy: the expert policy (doesn't exist in the form of an expert *policy* in our case
                    I'm just setting ac to None to get the expert policy! Maybe I can specify the expert policy as rule_based
                    and have elif statement
    collect_expert_data: whether or not I want to collect the expert's data i.e use the expert to *run* the env and collect data 
    I see: usually, the expert data is collect elsewhere and loaded in as a pkl file for training - it's called 'initial_expertdata'
    Also, because it'll be less efficient to reconstruct the env using only the state (?) I won't be using relabel with expert policy. 
    Instead, I'll be 'querying' user input while running DAgger.
    """

   ob = env.reset()
    # init vars
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:
        if render:
            env.render()
        
        obs.append(ob)
        assert len(env.controlled_vehicles) == 1
        controlled_vehicle = env.controlled_vehicles[0]

        # get collect policy to provide actions for stepping the env
        if isinstance(MLPPolicy, collect_policy):
            ac = collect_policy.get_action(ob)[0]
        else:
            print("collect_policy is always a MLPPolicySL for now; I’ll find another way to collect initial_expert_data")
            assert False

        # get expert policy to provide actions for training
        if expert_policy == 'rule_based':
            ac = controlled_vehicle.acceleration(controlled_vehicle)
            ac = ac.reshape(1, 1)
            acs.append(ac)
        else if expert_policy == 'human_input':
            print(f'expert policy is {expert_policy}; not implemented yet')
            assert False

        ob, rew, done, _ = env.step(ac)
        steps += 1
        next_obs.append(ob)
        rewards.append(rew)

        rollout_done = 1 if done or steps >= max_path_length else 0 # HINT: this is either 0 or 1
        terminals.append(rollout_done)

        if rollout_done:
            break
    
    return Path(obs, image_obs, acs, rewards, next_obs, terminals)


def sample_trajectories(env, collect_policy, min_timesteps_per_batch, max_path_length, render=False, expert_policy=None, collect_expert_data=False):
    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:
        path = sample_trajectory(env, collect_policy, max_path_length, expert_policy, collect_expert_data, render=render)
        timesteps_this_batch += get_pathlength(path)
        paths.append(path)
    return paths, timesteps_this_batch


############################################
############################################

def Path(obs, image_obs, acs, rewards, next_obs, terminals):
    """
        Take info (separate arrays) from a single rollout
        and return it in a single dictionary
    """
    if image_obs != []:
        image_obs = np.stack(image_obs, axis=0)
    return {"observation" : np.array(obs, dtype=np.float32),
            "image_obs" : np.array(image_obs, dtype=np.uint8),
            "reward" : np.array(rewards, dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)}


def convert_listofrollouts(paths, concat_rew=True):
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    if concat_rew:
        rewards = np.concatenate([path["reward"] for path in paths])
    else:
        rewards = [path["reward"] for path in paths]
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    return observations, actions, rewards, next_observations, terminals

############################################
############################################

def get_pathlength(path):
    return len(path["reward"])