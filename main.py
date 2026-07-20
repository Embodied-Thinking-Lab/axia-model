import gymnasium as gym
import panda_gym
import torch

from data_collection import build_transitions
from model import ReachModel

def main():
    num_steps = 100
    log_interval = 10
    
    env = gym.make("PandaReach-v3", render_mode="rgb_array", renderer="Tiny")
    obs, info = env.reset()
    model = ReachModel(state_dim=12, action_dim=3, latent_dim=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    
    img = env.render()

    for step in range(num_steps):
        
    
    
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, next_info = env.step(action)

        transition = build_transitions(obs, action, next_obs, info, next_info)

        outputs = model(transition["state"], transition["action"])

        with torch.no_grad():
            target_next_z = model.encode(transition["next_state"]).detach()
    
        goal_distance_loss = torch.nn.functional.mse_loss(outputs["goal_distance"], transition["goal_distance"])
        success_loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs["success_logit"], transition["success"])
        dynamics_loss = torch.nn.functional.mse_loss(outputs["next_z"], target_next_z)
        
        total_loss = goal_distance_loss+success_loss+dynamics_loss
    
    
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if terminated or truncated:
            obs, info = env.reset()
        else:
            obs, info = next_obs, next_info 

        if step % log_interval == 0:
            
            print(
                "step:", step,
                "\ndistance loss:", goal_distance_loss,
                "\nsucess loss:", success_loss,
                "\ndynamics loss:", dynamics_loss,
                "\ntotal loss:",total_loss,
            )
        
    


main()