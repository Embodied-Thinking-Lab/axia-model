import torch 
import torch.nn as nn

class StateEncoder(nn.Module):
    def __init__(self, state_dim, latent_dim=128):
        super().__init__()
        self.network = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.ELU(),
                nn.Linear(256, 256),
                nn.ELU(),
                nn.Linear(256, latent_dim)
        )

    def forward(self, state):
        return self.network(state)
        



class DynamicsModel(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim+action_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, z, action):
        x = torch.cat([z, action], dim=-1)
        return self.network(x)


class GoalDistanceHead(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ELU(),
            nn.Linear(128, 1)
        )

    def forward(self, z):
        return self.network(z)


class SuccessHead(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.network =  nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ELU(), 
            nn.Linear(128, 1)
        )

    def forward(self, z):
        return self.network(z)


class ReachModel(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim=128):
        super().__init__()

        self.encoder = StateEncoder(state_dim, latent_dim)
        self.dynamics = DynamicsModel(latent_dim, action_dim)
        self.goal_distance_head = GoalDistanceHead(latent_dim)
        self.success_head = SuccessHead(latent_dim)


    def encode(self, state):
        return self.encoder(state)

    def predict_next_latent(self, z, action):
        return self.dynamics(z, action)

    def predict_heads(self, z):
        return self.goal_distance_head(z), self.success_head(z)


    def forward(self, state, action=None):
        z = self.encode(state)
        goal_distance, success_logit = self.predict_heads(z)

        if (action is None):
            return {"z":z, "goal_distance": goal_distance, "success_logit": success_logit}

        else:
            next_z = self.predict_next_latent(z, action)
            return {"z":z, "goal_distance": goal_distance, "success_logit": success_logit, "next_z":next_z}


        
            