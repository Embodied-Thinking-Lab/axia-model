import gymnasium as gym
import panda_gym
import torch

from data_collection import ReplayBuffer, build_transitions
from model import ReachModel


def main():
    num_steps = 3000
    log_interval = 100
    eval_interval = 250
    checkpoint_interval = 500
    batch_size = 16
    replay_capacity = 1000
    warmup_steps = 32
    running_window = 100

    goal_distance_weight = 1.0
    success_weight = 0.1
    dynamics_weight = 0.5

    env = gym.make("PandaReach-v3", render_mode="rgb_array", renderer="Tiny")
    obs, info = env.reset()

    model = ReachModel(state_dim=12, action_dim=3, latent_dim=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    replay_buffer = ReplayBuffer(replay_capacity)

    distance_losses = []
    success_losses = []
    dynamics_losses = []
    total_losses = []
    success_targets = []

    env.render()

    for step in range(num_steps):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, next_info = env.step(action)

        transition = build_transitions(obs, action, next_obs, info, next_info)
        replay_buffer.add(transition)

        if len(replay_buffer) < batch_size:
            train_batch = transition
        else:
            train_batch = replay_buffer.sample(batch_size)

        outputs = model(train_batch["state"], train_batch["action"])

        with torch.no_grad():
            target_next_z = model.encode(train_batch["next_state"])

        goal_distance_loss = torch.nn.functional.mse_loss(
            outputs["goal_distance"], train_batch["goal_distance"]
        )
        success_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            outputs["success_logit"], train_batch["success"]
        )
        dynamics_loss = torch.nn.functional.mse_loss(outputs["next_z"], target_next_z)

        total_loss = (
            goal_distance_weight * goal_distance_loss
            + success_weight * success_loss
            + dynamics_weight * dynamics_loss
        )

        if step >= warmup_steps:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        distance_losses.append(goal_distance_loss.item())
        success_losses.append(success_loss.item())
        dynamics_losses.append(dynamics_loss.item())
        total_losses.append(total_loss.item())
        success_targets.append(train_batch["success"].mean().item())

        if len(distance_losses) > running_window:
            distance_losses.pop(0)
            success_losses.pop(0)
            dynamics_losses.pop(0)
            total_losses.pop(0)
            success_targets.pop(0)

        if terminated or truncated:
            obs, info = env.reset()
        else:
            obs, info = next_obs, next_info

        if step % log_interval == 0:
            mean_distance_loss = sum(distance_losses) / len(distance_losses)
            mean_success_loss = sum(success_losses) / len(success_losses)
            mean_dynamics_loss = sum(dynamics_losses) / len(dynamics_losses)
            mean_total_loss = sum(total_losses) / len(total_losses)
            mean_success_target = sum(success_targets) / len(success_targets)

            print(
                "step:", step,
                "\nreplay size:", len(replay_buffer),
                "\ndistance loss:", goal_distance_loss.item(),
                "\nsuccess loss:", success_loss.item(),
                "\ndynamics loss:", dynamics_loss.item(),
                "\ntotal loss:", total_loss.item(),
                "\nmean distance loss:", mean_distance_loss,
                "\nmean success loss:", mean_success_loss,
                "\nmean dynamics loss:", mean_dynamics_loss,
                "\nmean total loss:", mean_total_loss,
                "\nmean success target:", mean_success_target,
            )

        if step % eval_interval == 0:
            with torch.no_grad():
                eval_outputs = model(transition["state"], transition["action"])
                predicted_distance = eval_outputs["goal_distance"].mean().item()
                true_distance = transition["goal_distance"].mean().item()
                predicted_success = torch.sigmoid(eval_outputs["success_logit"]).mean().item()
                true_success = transition["success"].mean().item()

                print(
                    "eval step:", step,
                    "\npredicted distance:", predicted_distance,
                    "\ntrue distance:", true_distance,
                    "\npredicted success:", predicted_success,
                    "\ntrue success:", true_success,
                )

        if step > 0 and step % checkpoint_interval == 0:
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                f"reach_model_step_{step}.pt",
            )

    torch.save(
        {
            "step": num_steps,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        "reach_model.pt",
    )

main()
