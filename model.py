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