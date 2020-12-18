import numpy as np
import time

def sample_trajectory(env, collect_policy, max_path_length, render, expert_policy, collect_expert_data):
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
        ac = controlled_vehicle.acceleration(controlled_vehicle)
        # assert ac.shape is something?
        acs.append(ac.reshape(1, 1))

        ob, rew, done, _ = env.step(None)
        steps += 1
        next_obs.append(ob)
        rewards.append(rew)

        # TODO end the rollout if the rollout ended
        # HINT: rollout can end due to done, or due to max_path_length
        rollout_done = 1 if done or steps >= max_path_length else 0 # HINT: this is either 0 or 1
        terminals.append(rollout_done)

        if rollout_done:
            break
    
    return Path(obs, image_obs, acs, rewards, next_obs, terminals)


def sample_trajectories(env, collect_policy, min_timesteps_per_batch, max_path_length, render=False, expert_policy=None, collect_expert_data=False):
    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:
        path = sample_trajectory(env, collect_policy, max_path_length, render, expert_policy, collect_expert_data)
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