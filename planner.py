import torch


class Planner:
    def __init__(self, model, num_actions, horizon=1):
        self.model = model
        self.num_actions = num_actions
        self.horizon = horizon

    def _score_latents(self, rollout_z, objective):
        predictions = self.model.predict_heads(rollout_z)
        score = torch.zeros(rollout_z.shape[0], 1, device=rollout_z.device)

        goal_distance_target = objective.get("goal_distance")
        if goal_distance_target is not None:
            score = score + torch.abs(predictions["goal_distance"] - goal_distance_target)

        gripper_distance_target = objective.get("gripper_to_object_distance")
        if gripper_distance_target is not None:
            score = score + torch.abs(
                predictions["gripper_to_object_distance"] - gripper_distance_target
            )

        object_height_min = objective.get("object_height_min")
        if object_height_min is not None:
            score = score + torch.relu(object_height_min - predictions["object_height"])

        success_target = objective.get("success")
        if success_target is not None:
            success_prob = torch.sigmoid(predictions["success_logit"])
            score = score + torch.abs(success_prob - success_target)

        return score
        
    def select_action(self, state_batch, action_low, action_high, objective, action_mean=None):
        z = self.model.encode(state_batch)

        action_dim = action_low.shape[0]
        rand_actions = torch.rand(self.num_actions, self.horizon, action_dim, device=z.device)
        candidate_action_sequences = action_low + (action_high - action_low) * rand_actions

        if action_mean is not None:
            action_mean = action_mean.to(z.device)
            noise = 0.35 * torch.randn_like(candidate_action_sequences)
            candidate_action_sequences = action_mean[None, None, :] + noise
            candidate_action_sequences = torch.max(
                torch.min(candidate_action_sequences, action_high), action_low
            )
            candidate_action_sequences[0, :, :] = action_mean[None, :].repeat(self.horizon, 1)

        rollout_z = z.repeat(self.num_actions, 1)

        for horizon_step in range(self.horizon):
            rollout_actions = candidate_action_sequences[:, horizon_step, :]
            rollout_z = self.model.predict_next_latent(rollout_z, rollout_actions)

        predicted_distances = self._score_latents(rollout_z, objective)

        min_idx = predicted_distances.squeeze(-1).argmin()
        
        return candidate_action_sequences[min_idx, 0, :]
        

        
        

    
        
    
