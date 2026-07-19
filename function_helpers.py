import torch

def compute_goal_distance(obs):
    achieved_goal = torch.from_numpy(obs["achieved_goal"])
    desired_goal = torch.from_numpy(obs["desired_goal"])
    
    goal_distance = torch.norm(achieved_goal - desired_goal)
    return goal_distance
    
def flatten_observation(obs):
    # observation is a dict with numpy arrays
    np_obs_list = [obs["observation"], obs["achieved_goal"], obs["desired_goal"]]

    # turn the numpy arrays to torch tensors
    tensor_list = [torch.from_numpy(arr) for arr in np_obs_list]
    state = torch.cat(tensor_list)

    return state
