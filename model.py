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


class ObjectHeightHead(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ELU(),
            nn.Linear(128, 1)
        )

    def forward(self, z):
        return self.network(z)


class GripperToObjectDistanceHead(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ELU(),
            nn.Linear(128, 1)
        )

    def forward(self, z):
        return self.network(z)


class RecoveryActorHead(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ELU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.network(z)


class WorldModel(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim=128):
        super().__init__()

        self.encoder = StateEncoder(state_dim, latent_dim)
        self.dynamics = DynamicsModel(latent_dim, action_dim)
        self.goal_distance_head = GoalDistanceHead(latent_dim)
        self.success_head = SuccessHead(latent_dim)
        self.object_height_head = ObjectHeightHead(latent_dim)
        self.gripper_to_object_distance_head = GripperToObjectDistanceHead(latent_dim)
        self.recovery_actor_head = RecoveryActorHead(latent_dim, action_dim)


    def encode(self, state):
        return self.encoder(state)

    def predict_next_latent(self, z, action):
        return self.dynamics(z, action)

    def predict_heads(self, z):
        return {
            "goal_distance": self.goal_distance_head(z),
            "success_logit": self.success_head(z),
            "object_height": self.object_height_head(z),
            "gripper_to_object_distance": self.gripper_to_object_distance_head(z),
            "recovery_action": self.recovery_actor_head(z),
        }

    def forward(self, state, action=None):
        z = self.encode(state)
        head_outputs = self.predict_heads(z)
        outputs = {"z": z, **head_outputs}

        if (action is None):
            return outputs

        else:
            next_z = self.predict_next_latent(z, action)
            outputs["next_z"] = next_z
            return outputs


        
            
