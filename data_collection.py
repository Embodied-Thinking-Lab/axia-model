import random

import torch 
from function_helpers import (
    compute_goal_distance,
    compute_gripper_to_object_distance,
    compute_object_height,
    compute_recovery_action_target,
    flatten_observation,
)

def build_targets(obs, info, action_dim, recovery_action_target=None):
    
    goal_distance = compute_goal_distance(obs)
    goal_distance_float = goal_distance[None, None, ...]
    goal_distance_float = goal_distance_float.to(torch.float32)

    object_height = compute_object_height(obs)
    object_height_float = object_height[None, None, ...]
    object_height_float = object_height_float.to(torch.float32)

    gripper_to_object_distance = compute_gripper_to_object_distance(obs)
    gripper_to_object_distance_float = gripper_to_object_distance[None, None, ...]
    gripper_to_object_distance_float = gripper_to_object_distance_float.to(torch.float32)

    if recovery_action_target is None:
        recovery_action_target = compute_recovery_action_target(obs, action_dim)
    recovery_action_target = recovery_action_target[None, ...]

    env_success = info["is_success"]
    env_success = torch.tensor(env_success).to(torch.float32)
    env_success_float = env_success[None, None, ...]

    return {
        "goal_distance": goal_distance_float,
        "object_height": object_height_float,
        "gripper_to_object_distance": gripper_to_object_distance_float,
        "recovery_action_target": recovery_action_target,
        "success": env_success_float,
    }


def build_transitions(
    obs,
    action,
    next_obs,
    info,
    next_info,
    recovery_supervision_mask=0.0,
    recovery_action_target=None,
):
    action_dim = action.shape[0]
    targets = build_targets(obs, info, action_dim, recovery_action_target=recovery_action_target)
    next_targets = build_targets(next_obs, next_info, action_dim)

    action_tensor = torch.from_numpy(action)
    action_batch = torch.unsqueeze(action_tensor, 0)
    
    state = flatten_observation(obs)
    state_batch = torch.unsqueeze(state, 0)

    next_state = flatten_observation(next_obs)
    next_state_batch = torch.unsqueeze(next_state, 0)
    recovery_mask = torch.tensor([[recovery_supervision_mask]], dtype=torch.float32)
    
    
    return {
        "state": state_batch,
        "action": action_batch, 
        "next_state": next_state_batch,
        "goal_distance": targets["goal_distance"],
        "next_goal_distance": next_targets["goal_distance"],
        "object_height": targets["object_height"],
        "next_object_height": next_targets["object_height"],
        "gripper_to_object_distance": targets["gripper_to_object_distance"],
        "next_gripper_to_object_distance": next_targets["gripper_to_object_distance"],
        "recovery_action_target": targets["recovery_action_target"],
        "recovery_supervision_mask": recovery_mask,
        "success": targets["success"],
        "next_success": next_targets["success"],
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
            "next_goal_distance": torch.cat([transition["next_goal_distance"] for transition in batch], dim=0),
            "object_height": torch.cat([transition["object_height"] for transition in batch], dim=0),
            "next_object_height": torch.cat([transition["next_object_height"] for transition in batch], dim=0),
            "gripper_to_object_distance": torch.cat([transition["gripper_to_object_distance"] for transition in batch], dim=0),
            "next_gripper_to_object_distance": torch.cat([transition["next_gripper_to_object_distance"] for transition in batch], dim=0),
            "recovery_action_target": torch.cat([transition["recovery_action_target"] for transition in batch], dim=0),
            "recovery_supervision_mask": torch.cat([transition["recovery_supervision_mask"] for transition in batch], dim=0),
            "success": torch.cat([transition["success"] for transition in batch], dim=0),
            "next_success": torch.cat([transition["next_success"] for transition in batch], dim=0),
        }

    def __len__(self):
        return len(self.buffer)
