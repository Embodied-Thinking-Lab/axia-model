import time

import gymnasium as gym
import numpy as np
import panda_gym
import torch
from gymnasium.wrappers import TimeLimit

from function_helpers import *
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


def get_recovery_diagnostics(obs, approach_height):
    gripper_pos = obs["observation"][:3]
    object_pos = obs["achieved_goal"]

    xy_distance = np.linalg.norm(gripper_pos[:2] - object_pos[:2])
    target_z = object_pos[2] + approach_height
    z_distance = abs(gripper_pos[2] - target_z)
    gripper_to_object_distance = np.linalg.norm(gripper_pos - object_pos)

    return xy_distance, z_distance, gripper_to_object_distance


def apply_object_perturbation(env, object_offset_xy):
    unwrapped_env = env.unwrapped
    nominal_object_position = np.array(
        unwrapped_env.task.sim.get_base_position("object"), dtype=np.float32
    )

    perturbed_position = nominal_object_position.copy()
    perturbed_position[:2] += object_offset_xy
    unwrapped_env.task.sim.set_base_pose(
        "object",
        perturbed_position,
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )

    obs = unwrapped_env._get_obs()
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


def make_env(env_name, render_mode, max_episode_steps):
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

    state_batch = torch.unsqueeze(flatten_observation(obs), 0)
    action_low = torch.as_tensor(env.action_space.low, dtype=torch.float32)
    action_high = torch.as_tensor(env.action_space.high, dtype=torch.float32)

    with torch.no_grad():
        action = planner.select_action(
            state_batch,
            action_low,
            action_high,
            objective=objective,
            action_mean=torch.from_numpy(teacher_action).to(torch.float32),
        )

    action = action.cpu().numpy()
    if action.shape[0] > 3:
        action[3] = teacher_action[3]
    return action


def main():
    env_name = "PandaPickAndPlace-v3"
    render_mode = "human" 
    max_total_steps = 500
    max_episodes = 5
    # step_delay = 0.08
    step_delay = 0.10
    end_pause_seconds = 5.0
    max_episode_steps = 150
    perturb_object = True
    object_offset_xy = np.array([0.04, 0.04], dtype=np.float32)
    max_recovery_steps = 50

    env = make_env(env_name, render_mode, max_episode_steps=max_episode_steps)
    initial_obs, initial_info = env.reset()

    state_dim = flatten_observation(initial_obs).shape[0]
    action_dim = env.action_space.shape[0]

    model = WorldModel(state_dim=state_dim, action_dim=action_dim, latent_dim=128)
    checkpoint = torch.load("model_files/pick_and_place_model.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    planner = Planner(model, num_actions=256, horizon=6)

    policy = ScriptedPickAndPlacePolicy(action_dim=action_dim)
    recovery_policy = ScriptedRecoveryPolicy(action_dim=action_dim)

    obs, info = reset_episode(env, policy, perturb_object, object_offset_xy)

    if render_mode == "human":
        env.render()

    recovery_mode = False
    recovery_steps = 0
    episode_idx = 0
    episode_step = 0
    total_successes = 0
    failed_grasp_events = 0
    recovery_attempts = 0
    recovery_successes = 0
    recovery_timeouts = 0
    post_recovery_successes = 0
    post_recovery_failed_retries = 0
    pending_post_recovery = False

    for step in range(max_total_steps):
        episode_step += 1

        if recovery_mode:
            action = select_recovery_action(model, planner, recovery_policy, obs, env)
            controller_name = "planned_recovery"
        else:
            action = policy.act(obs)
            controller_name = "scripted"

        next_obs, reward, terminated, truncated, next_info = env.step(action)
        goal_distance = compute_goal_distance(next_obs).item()
        recovery_xy_distance, recovery_z_distance, recovery_distance = get_recovery_diagnostics(
            next_obs, policy.approach_height
        )

        if not recovery_mode and policy.failed_grasp:
            recovery_mode = True
            recovery_steps = 0
            recovery_policy.reset()
            failed_grasp_events += 1
            recovery_attempts += 1
            print("failed grasp detected, switching to planned recovery")

        if recovery_mode:
            recovery_steps += 1

            if is_recovered(next_obs):
                actual_object_position = np.array(next_obs["achieved_goal"], dtype=np.float32)
                policy.set_expected_object_pos(actual_object_position)
                policy.clear_failed_grasp()
                policy.set_phase("move_above_goal")
                recovery_mode = False
                recovery_steps = 0
                recovery_policy.reset()
                recovery_successes += 1
                pending_post_recovery = True
                print("planned recovery succeeded, resuming scripted routine")

            elif recovery_steps >= max_recovery_steps:
                recovery_mode = False
                recovery_steps = 0
                recovery_timeouts += 1
                policy.clear_failed_grasp()
                policy.set_phase("move_above_object")
                recovery_policy.reset()
                print("planned recovery timed out, resetting scripted retry")

        print(
            "----------------",
            "step:", step,
            "\nepisode:", episode_idx,
            "\nepisode_step:", episode_step,
            "\ncontroller:", controller_name,
            "\nphase:", policy.phase,
            "\nrecovery_mode:", recovery_mode,
            "\nrecovery_steps:", recovery_steps,
            "\nreward:", reward,
            "\ngoal distance:", goal_distance,
            "\nis_success:", next_info["is_success"],
            "\nfailed_grasp:", policy.failed_grasp,
            "\nrecovery_xy_distance:", recovery_xy_distance,
            "\nrecovery_z_distance:", recovery_z_distance,
            "\nrecovery_distance:", recovery_distance,
            "\nfailed_grasp_events:", failed_grasp_events,
            "\nrecovery_attempts:", recovery_attempts,
            "\nrecovery_successes:", recovery_successes,
            "\nrecovery_timeouts:", recovery_timeouts,
            "\ntotal_successes:", total_successes,
        )

        if render_mode == "human":
            env.render()
            time.sleep(step_delay)

        if next_info["is_success"]:
            total_successes += 1
            if pending_post_recovery:
                post_recovery_successes += 1
                pending_post_recovery = False
            print("corrective routine reached success, starting next perturbed episode")
            episode_idx += 1
            if episode_idx >= max_episodes:
                break

            obs, info = reset_episode(env, policy, perturb_object, object_offset_xy)
            recovery_mode = False
            recovery_steps = 0
            recovery_policy.reset()
            episode_step = 0
            continue

        if terminated or truncated:
            if recovery_mode:
                recovery_timeouts += 1
                recovery_mode = False
                recovery_steps = 0
                recovery_policy.reset()

            if pending_post_recovery:
                post_recovery_failed_retries += 1
                pending_post_recovery = False

            print("episode ended, starting next perturbed episode")
            episode_idx += 1
            if episode_idx >= max_episodes:
                break

            obs, info = reset_episode(env, policy, perturb_object, object_offset_xy)
            recovery_mode = False
            recovery_steps = 0
            recovery_policy.reset()
            episode_step = 0
            continue

        obs, info = next_obs, next_info

        if pending_post_recovery and policy.failed_grasp:
            post_recovery_failed_retries += 1
            pending_post_recovery = False

    print(
        "================",
        "\nfinished corrective evaluation",
        "\nepisodes run:", episode_idx,
        "\ntotal successes:", total_successes,
        "\nfailed grasp events:", failed_grasp_events,
        "\nrecovery attempts:", recovery_attempts,
        "\nrecovery successes:", recovery_successes,
        "\npost recovery successes:", post_recovery_successes,
        "\npost recovery failed retries:", post_recovery_failed_retries,
        "\nrecovery timeouts:", recovery_timeouts,
    )
    env.close()
    if render_mode == "human":
        time.sleep(end_pause_seconds)


if __name__ == "__main__":
    main()
