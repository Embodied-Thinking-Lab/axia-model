import gymnasium as gym
import panda_gym
import torch

from data_collection import build_transitions
from model import ReachModel

def main():
    env = gym.make("PandaReach-v3", render_mode="rgb_array", renderer="Tiny")
    obs, info = env.reset()
    
    img = env.render()
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, next_info = env.step(action)

    transition = build_transitions(obs, action, next_obs, info, next_info)

    model = ReachModel(state_dim=12, action_dim=3, latent_dim=128)

    outputs = model(transition["state"], transition["action"])



main()