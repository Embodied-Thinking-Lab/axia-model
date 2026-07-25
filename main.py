import gymnasium as gym
import numpy as np
import panda_gym
import torch
from gymnasium.wrappers import TimeLimit

from data_collection import ReplayBuffer, build_transitions
from function_helpers import (
    compute_gripper_to_object_distance,
    compute_object_height,
    flatten_observation,
)
from model import WorldModel
from planner import Planner
from scripted_pick_and_place import ScriptedPickAndPlacePolicy, ScriptedRecoveryPolicy


def is_recovered(obs, min_object_height=0.03, max_gripper_to_object_distance=0.04):
    object_height = compute_object_height(obs).item()
    gripper_to_object_distance = compute_gripper_to_object_distance(obs).item()

    return (
        object_height > min_object_height
        and gripper_to_object_distance < max_gripper_to_object_distance
    )


def apply_object_perturbation(env, object_offset_xy):
    nominal_object_position = np.array(
        env.unwrapped.task.sim.get_base_position("object"), dtype=np.float32
    )
    perturbed_position = nominal_object_position.copy()
    perturbed_position[:2] += object_offset_xy
    env.unwrapped.task.sim.set_base_pose(
        "object",
        perturbed_position,
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )
    obs = env.unwrapped._get_obs()
    return obs, nominal_object_position


def reset_episode(env, policy, perturb_object, object_offset_xy):
    obs, info = env.reset()

    nominal_object_position = np.array(
        env.unwrapped.task.sim.get_base_position("object"), dtype=np.float32
    )

    if perturb_object:
        obs, nominal_object_position = apply_object_perturbation(env, object_offset_xy)

    policy.reset()
    policy.set_expected_object_pos(nominal_object_position)
    return obs, info


def make_env(env_name, max_episode_steps, render_mode="rgb_array"):
    base_env = gym.make(env_name, render_mode=render_mode, renderer="Tiny")
    return TimeLimit(base_env.unwrapped, max_episode_steps=max_episode_steps)


def select_recovery_action(model, planner, recovery_policy, obs, env):
    teacher_action = recovery_policy.act(obs)
    phase = recovery_policy.phase

    if phase in {"move_above_object", "descend_to_object", "close_gripper"}:
        return teacher_action

    objective = {
        "gripper_to_object_distance": torch.tensor([[0.0]]),
        "object_height_min": torch.tensor([[0.04]]),
    }

    state = flatten_observation(obs).unsqueeze(0)
    action_low = torch.as_tensor(env.action_space.low, dtype=torch.float32)
    action_high = torch.as_tensor(env.action_space.high, dtype=torch.float32)

    with torch.no_grad():
        planned_action = planner.select_action(
            state,
            action_low,
            action_high,
            objective=objective,
            action_mean=torch.from_numpy(teacher_action).to(torch.float32),
        )

    action = planned_action.cpu().numpy()
    if action.shape[0] > 3:
        action[3] = teacher_action[3]

    return action


def run_rollout_eval(model, env_name, perturb_object, object_offset_xy, max_episode_steps, episodes=3):
    env = make_env(env_name, max_episode_steps=max_episode_steps)
    action_dim = env.action_space.shape[0]
    policy = ScriptedPickAndPlacePolicy(action_dim=action_dim)
    recovery_policy = ScriptedRecoveryPolicy(action_dim=action_dim)
    planner = Planner(model, num_actions=256, horizon=6)

    successes = 0
    recoveries = 0

    for _ in range(episodes):
        obs, info = reset_episode(env, policy, perturb_object, object_offset_xy)
        recovery_mode = False

        for _ in range(max_episode_steps):
            if recovery_mode:
                action = select_recovery_action(model, planner, recovery_policy, obs, env)
            else:
                action = policy.act(obs)

            next_obs, _, terminated, truncated, next_info = env.step(action)

            if not recovery_mode and policy.failed_grasp:
                recovery_mode = True
                recovery_policy.reset()

            if recovery_mode:
                if recovery_policy.phase == "lift_object" and is_recovered(next_obs):
                    actual_object_position = np.array(next_obs["achieved_goal"], dtype=np.float32)
                    policy.set_expected_object_pos(actual_object_position)
                    policy.clear_failed_grasp()
                    policy.set_phase("move_above_goal")
                    recovery_mode = False
                    recoveries += 1

            if next_info["is_success"]:
                successes += 1
                break

            if terminated or truncated:
                break

            obs, info = next_obs, next_info

    env.close()
    return successes / episodes, recoveries / episodes


def main():
    env_name = "PandaPickAndPlace-v3"
    num_steps = 3000
    log_interval = 100
    eval_interval = 250
    checkpoint_interval = 500
    max_episode_steps = 150
    batch_size = 16
    replay_capacity = 1000
    warmup_steps = 32
    running_window = 100
    perturb_object = True
    object_offset_xy = np.array([0.04, 0.04], dtype=np.float32)
    max_recovery_steps = 40

    goal_distance_weight = 1.0
    success_weight = 0.1
    dynamics_weight = 0.5
    predicted_head_weight = 1.0
    object_height_weight = 0.5
    gripper_to_object_distance_weight = 0.5
    recovery_action_weight = 1.0

    env = make_env(env_name, max_episode_steps=max_episode_steps)
    obs, info = env.reset()

    state_dim = flatten_observation(obs).shape[0]
    action_dim = env.action_space.shape[0]

    model = WorldModel(state_dim=state_dim, action_dim=action_dim, latent_dim=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    replay_buffer = ReplayBuffer(replay_capacity)
    policy = ScriptedPickAndPlacePolicy(action_dim=action_dim)
    recovery_policy = ScriptedRecoveryPolicy(action_dim=action_dim)

    obs, info = reset_episode(env, policy, perturb_object, object_offset_xy)

    distance_losses = []
    object_height_losses = []
    gripper_to_object_distance_losses = []
    recovery_action_losses = []
    success_losses = []
    dynamics_losses = []
    total_losses = []
    success_targets = []
    recovery_mask_means = []

    env.render()

    recovery_mode = False
    recovery_steps = 0
    recovery_teacher_steps = 0
    recovery_teacher_successes = 0
    recovery_teacher_timeouts = 0

    for step in range(num_steps):
        recovery_supervision_mask = 0.0
        recovery_action_target = None

        if recovery_mode:
            action = recovery_policy.act(obs)
            recovery_action_target = torch.from_numpy(action).to(torch.float32)
            recovery_supervision_mask = 1.0
        else:
            action = policy.act(obs)

        next_obs, reward, terminated, truncated, next_info = env.step(action)

        transition = build_transitions(
            obs,
            action,
            next_obs,
            info,
            next_info,
            recovery_supervision_mask=recovery_supervision_mask,
            recovery_action_target=recovery_action_target,
        )
        replay_buffer.add(transition)

        if not recovery_mode and policy.failed_grasp:
            recovery_mode = True
            recovery_steps = 0
            recovery_policy.reset()

        if recovery_mode:
            recovery_steps += 1
            recovery_teacher_steps += 1

            if is_recovered(next_obs):
                actual_object_position = np.array(next_obs["achieved_goal"], dtype=np.float32)
                policy.set_expected_object_pos(actual_object_position)
                policy.clear_failed_grasp()
                policy.set_phase("move_above_goal")
                recovery_mode = False
                recovery_steps = 0
                recovery_policy.reset()
                recovery_teacher_successes += 1

            elif recovery_steps >= max_recovery_steps:
                policy.clear_failed_grasp()
                policy.set_phase("move_above_object")
                recovery_mode = False
                recovery_steps = 0
                recovery_policy.reset()
                recovery_teacher_timeouts += 1

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
        object_height_loss = torch.nn.functional.mse_loss(
            outputs["object_height"], train_batch["object_height"]
        )
        gripper_to_object_distance_loss = torch.nn.functional.mse_loss(
            outputs["gripper_to_object_distance"], train_batch["gripper_to_object_distance"]
        )
        recovery_action_error = (outputs["recovery_action"] - train_batch["recovery_action_target"]) ** 2
        recovery_action_error = recovery_action_error.mean(dim=-1, keepdim=True)
        recovery_mask = train_batch["recovery_supervision_mask"]
        if recovery_mask.sum().item() > 0:
            recovery_action_loss = (recovery_action_error * recovery_mask).sum() / recovery_mask.sum()
        else:
            recovery_action_loss = torch.zeros((), device=outputs["recovery_action"].device)
        success_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            outputs["success_logit"], train_batch["success"]
        )
        dynamics_loss = torch.nn.functional.mse_loss(outputs["next_z"], target_next_z)
        predicted_next_heads = model.predict_heads(outputs["next_z"])
        next_goal_distance_loss = torch.nn.functional.mse_loss(
            predicted_next_heads["goal_distance"], train_batch["next_goal_distance"]
        )
        next_object_height_loss = torch.nn.functional.mse_loss(
            predicted_next_heads["object_height"], train_batch["next_object_height"]
        )
        next_gripper_to_object_distance_loss = torch.nn.functional.mse_loss(
            predicted_next_heads["gripper_to_object_distance"],
            train_batch["next_gripper_to_object_distance"],
        )
        next_success_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            predicted_next_heads["success_logit"], train_batch["next_success"]
        )

        total_loss = (
            goal_distance_weight * goal_distance_loss
            + object_height_weight * object_height_loss
            + gripper_to_object_distance_weight * gripper_to_object_distance_loss
            + recovery_action_weight * recovery_action_loss
            + success_weight * success_loss
            + dynamics_weight * dynamics_loss
            + predicted_head_weight
            * (
                goal_distance_weight * next_goal_distance_loss
                + object_height_weight * next_object_height_loss
                + gripper_to_object_distance_weight * next_gripper_to_object_distance_loss
                + success_weight * next_success_loss
            )
        )

        if step >= warmup_steps:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        distance_losses.append(goal_distance_loss.item())
        object_height_losses.append(object_height_loss.item())
        gripper_to_object_distance_losses.append(gripper_to_object_distance_loss.item())
        recovery_action_losses.append(recovery_action_loss.item())
        success_losses.append(success_loss.item())
        dynamics_losses.append(dynamics_loss.item())
        total_losses.append(total_loss.item())
        success_targets.append(train_batch["success"].mean().item())
        recovery_mask_means.append(recovery_mask.mean().item())

        if len(distance_losses) > running_window:
            distance_losses.pop(0)
            object_height_losses.pop(0)
            gripper_to_object_distance_losses.pop(0)
            recovery_action_losses.pop(0)
            success_losses.pop(0)
            dynamics_losses.pop(0)
            total_losses.pop(0)
            success_targets.pop(0)
            recovery_mask_means.pop(0)

        if terminated or truncated:
            obs, info = reset_episode(env, policy, perturb_object, object_offset_xy)
            recovery_mode = False
            recovery_steps = 0
            recovery_policy.reset()
        else:
            obs, info = next_obs, next_info

        if step % log_interval == 0:
            mean_distance_loss = sum(distance_losses) / len(distance_losses)
            mean_object_height_loss = sum(object_height_losses) / len(object_height_losses)
            mean_gripper_to_object_distance_loss = sum(gripper_to_object_distance_losses) / len(gripper_to_object_distance_losses)
            mean_recovery_action_loss = sum(recovery_action_losses) / len(recovery_action_losses)
            mean_success_loss = sum(success_losses) / len(success_losses)
            mean_dynamics_loss = sum(dynamics_losses) / len(dynamics_losses)
            mean_total_loss = sum(total_losses) / len(total_losses)
            mean_success_target = sum(success_targets) / len(success_targets)
            mean_recovery_mask = sum(recovery_mask_means) / len(recovery_mask_means)

            print(
                "step:", step,
                "\nreplay size:", len(replay_buffer),
                "\ndistance loss:", goal_distance_loss.item(),
                "\nobject height loss:", object_height_loss.item(),
                "\ngripper to object distance loss:", gripper_to_object_distance_loss.item(),
                "\nrecovery action loss:", recovery_action_loss.item(),
                "\nsuccess loss:", success_loss.item(),
                "\ndynamics loss:", dynamics_loss.item(),
                "\ntotal loss:", total_loss.item(),
                "\nmean distance loss:", mean_distance_loss,
                "\nmean object height loss:", mean_object_height_loss,
                "\nmean gripper to object distance loss:", mean_gripper_to_object_distance_loss,
                "\nmean recovery action loss:", mean_recovery_action_loss,
                "\nmean success loss:", mean_success_loss,
                "\nmean dynamics loss:", mean_dynamics_loss,
                "\nmean total loss:", mean_total_loss,
                "\nmean success target:", mean_success_target,
                "\nmean recovery mask:", mean_recovery_mask,
                "\nrecovery teacher steps:", recovery_teacher_steps,
                "\nrecovery teacher successes:", recovery_teacher_successes,
                "\nrecovery teacher timeouts:", recovery_teacher_timeouts,
            )

        if step % eval_interval == 0:
            with torch.no_grad():
                eval_outputs = model(transition["state"], transition["action"])
                predicted_distance = eval_outputs["goal_distance"].mean().item()
                true_distance = transition["goal_distance"].mean().item()
                predicted_object_height = eval_outputs["object_height"].mean().item()
                true_object_height = transition["object_height"].mean().item()
                predicted_gripper_to_object_distance = eval_outputs["gripper_to_object_distance"].mean().item()
                true_gripper_to_object_distance = transition["gripper_to_object_distance"].mean().item()
                predicted_success = torch.sigmoid(eval_outputs["success_logit"]).mean().item()
                true_success = transition["success"].mean().item()
                rollout_success_rate, recovery_rate = run_rollout_eval(
                    model,
                    env_name,
                    perturb_object,
                    object_offset_xy,
                    max_episode_steps,
                    episodes=2,
                )

                print(
                    "eval step:", step,
                    "\npredicted distance:", predicted_distance,
                    "\ntrue distance:", true_distance,
                    "\npredicted object height:", predicted_object_height,
                    "\ntrue object height:", true_object_height,
                    "\npredicted gripper to object distance:", predicted_gripper_to_object_distance,
                    "\ntrue gripper to object distance:", true_gripper_to_object_distance,
                    "\npredicted success:", predicted_success,
                    "\ntrue success:", true_success,
                    "\nrollout success rate:", rollout_success_rate,
                    "\nrollout recovery rate:", recovery_rate,
                )

        if step > 0 and step % checkpoint_interval == 0:
            torch.save(
                {
                    "step": step,
                    "env_name": env_name,
                    "state_dim": state_dim,
                    "action_dim": action_dim,
                    "max_episode_steps": max_episode_steps,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                f"model_files/pick_and_place_model_step_{step}.pt",
            )

    torch.save(
        {
            "step": num_steps,
            "env_name": env_name,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "max_episode_steps": max_episode_steps,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        "model_files/pick_and_place_model.pt",
    )
    env.close()

if __name__ == "__main__":
    main()
