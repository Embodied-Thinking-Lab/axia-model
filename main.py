import gymnasium as gym
import panda_gym
import torch

from model import DynamicsModel, GoalDistanceHead, StateEncoder, SuccessHead
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

    # add a new dimension for batch (B)
    state_batch = torch.unsqueeze(state, 0)

    state_encoder = StateEncoder(state_dim=12, latent_dim=128)
    latent_tensor = state_encoder(state_batch)

    action_tensor = torch.from_numpy(action)
    action_batch = torch.unsqueeze(action_tensor, 0)

    dynamics_model = DynamicsModel(latent_dim=128, action_dim=3)

    next_latent = dynamics_model(latent_tensor, action_batch)

    goal_dist_head = GoalDistanceHead(latent_dim=128)
    success_head = SuccessHead(latent_dim=128)

    goal_head_out = goal_dist_head(latent_tensor)
    success_head_out = success_head(latent_tensor)

    # targets 
    # the success target can be found in the info variable, its a bool

    goal_distance = compute_goal_distance(obs)

    print(goal_head_out)
    print(goal_distance)
    print(success_head_out)
    env_success = info["is_success"]
    print(env_success)




main()