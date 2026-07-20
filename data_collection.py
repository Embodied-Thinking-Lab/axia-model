import random

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


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def add(self, transition):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)

        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return {
            "state": torch.cat([transition["state"] for transition in batch], dim=0),
            "action": torch.cat([transition["action"] for transition in batch], dim=0),
            "next_state": torch.cat([transition["next_state"] for transition in batch], dim=0),
            "goal_distance": torch.cat([transition["goal_distance"] for transition in batch], dim=0),
            "success": torch.cat([transition["success"] for transition in batch], dim=0),
        }

    def __len__(self):
        return len(self.buffer)
