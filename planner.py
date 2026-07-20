import torch

class Planner:
    def __init__(self, model, num_actions):
        self.model = model
        self.num_actions = num_actions
        
    def select_action(self, state_batch, action_low, action_high):
        z = self.model.encode(state_batch)

        action_dim = action_low.shape[0]
        rand_actions = torch.rand(self.num_actions, action_dim)
        candidate_actions = action_low + (action_high - action_low) * rand_actions

        z_repeat = z.repeat(self.num_actions, 1)

        next_z = self.model.predict_next_latent(z_repeat, candidate_actions)
        predicted_distances = self.model.goal_distance_head(next_z)

        min_idx = predicted_distances.squeeze(-1).argmin()
        
        return candidate_actions[min_idx]
        

        
        

    
        
    
