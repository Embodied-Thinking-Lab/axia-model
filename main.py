import gymnasium as gym
import panda_gym
import torch

from model import ReachModel
from function_helpers import compute_goal_distance, flatten_observation


def main():
    env = gym.make("PandaReach-v3", render_mode="rgb_array", renderer="Tiny")
    obs, info = env.reset()
    
    # print("obs space:", env.observation_space)
    # print("action space:", env.action_space)
    # print("obs keys:", obs.keys() if isinstance(obs, dict) else "Not dict")
    
    img = env.render()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    state = flatten_observation(obs)

    model = ReachModel(state_dim=12, action_dim=3, latent_dim=128)

    # add a new dimension for batch (B)
    state_batch = torch.unsqueeze(state, 0)

    action_tensor = torch.from_numpy(action)
    action_batch = torch.unsqueeze(action_tensor, 0)

    outputs = model(state_batch, action_batch)
    
    goal_distance = compute_goal_distance(obs)
    env_success = info["is_success"]
    
    env_success = torch.tensor(env_success).to(torch.float32)
    
    goal_distance_float = goal_distance[None, None, ...]
    goal_distance_float = goal_distance_float.to(torch.float32)

    env_success_float = env_success[None, None, ...]



main()