import torch 
from function_helpers import compute_goal_distance, flatten_observation

def build_targets(obs, info):
    
    goal_distance = compute_goal_distance(obs)
    goal_distance_float = goal_distance[None, None, ...]
    goal_distance_float = goal_distance_float.to(torch.float32)

    env_success = info["is_success"]
    env_success = torch.tensor(env_success).to(torch.float32)
    env_success_float = env_success[None, None, ...]

    return {"goal_distance": goal_distance_float, "success": env_success_float}


def build_transitions(obs, action, next_obs, info, next_info):
    targets = build_targets(obs, info)

    action_tensor = torch.from_numpy(action)
    action_batch = torch.unsqueeze(action_tensor, 0)
    
    state = flatten_observation(obs)
    state_batch = torch.unsqueeze(state, 0)

    next_state = flatten_observation(next_obs)
    next_state_batch = torch.unsqueeze(next_state, 0)
    
    
    return {
        "state": state_batch,
        "action": action_batch, 
        "next_state": next_state_batch,
        "goal_distance": targets["goal_distance"],
        "success": targets["success"]
    }
