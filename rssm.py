import torch 
import torch.nn as nn
import math 

class RSSM(nn.module):
    def __init__(self, action_dim, embedding_dim, 
                 ht_dim=512, categoricals=32, classes=32):
        super().__init__()
        self.ht_dim = ht_dim
        self.categoricals = categoricals
        self.classes = classes
        self.zt_dim = categoricals*classes
        
        # sequence model
        self.gru_network = nn.GRUCell(
            input_size=self.zt_dim + action_dim,
            hidden_size=ht_dim
        ) 
        
        # representation model (called the encoder in the paper)
        self.representation_network = nn.Sequential(
            nn.Linear(self.ht_dim+self.embedding_dim, 1024),
            nn.LayerNorm(1024),
            nn.ELU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.ELU(),
            nn.Linear(1024, self.zt_dim)
        )
        
        # named dynamics predictor in paper
        self.transition_network = nn.Sequential(
            nn.Linear(self.ht_dim, 1024),
            nn.LayerNorm(1024),
            nn.ELU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.ELU(),
            nn.Linear(1024, self.zt_dim)
        )
        
        
        
        