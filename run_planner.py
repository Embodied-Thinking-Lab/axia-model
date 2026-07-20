import gymnasium as gym
import panda_gym
import torch
import time

from function_helpers import compute_goal_distance, flatten_observation
from model import ReachModel
from planner import Planner


def main():
    num_actions = 64
    max_steps = 200
    step_delay = 0.1
    end_pause_seconds = 5.0

    env = gym.make("PandaReach-v3", render_mode="human", renderer="Tiny")
    obs, info = env.reset()

    model = ReachModel(state_dim=12, action_dim=3, latent_dim=128)
    checkpoint = torch.load("model_files/reach_model.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    planner = Planner(model, num_actions=num_actions)

    action_low = torch.from_numpy(env.action_space.low).to(torch.float32)
    action_high = torch.from_numpy(env.action_space.high).to(torch.float32)

    env.render()

    success_step = None

    for step in range(max_steps):
        state = flatten_observation(obs)
        state_batch = torch.unsqueeze(state, 0)

        with torch.no_grad():
            action = planner.select_action(state_batch, action_low, action_high)

        action_numpy = action.cpu().numpy()
        next_obs, reward, terminated, truncated, next_info = env.step(action_numpy)

        goal_distance = compute_goal_distance(next_obs).item()

        print(
            "step:", step,
            "\nreward:", reward,
            "\ngoal distance:", goal_distance,
            "\nis_success:", next_info["is_success"],
        )

        env.render()
        time.sleep(step_delay)

        if next_info["is_success"]:
            print("success reached, pausing before exit...")
            time.sleep(end_pause_seconds)
            break

        if terminated or truncated:
            print("episode ended, pausing before exit...")
            time.sleep(end_pause_seconds)
            break

        obs, info = next_obs, next_info


main()
