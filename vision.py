import torch 
import torch.nn as nn
import math 

class Encoder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 4, 2)
        self.elu = nn.ELU()
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 128, 4, 2)
        self.conv4 = nn.Conv2d(128, 256, 4, 2)
        self.flatten = nn.Flatten(start_dim=1)
        self.linear = nn.Linear(4096, embedding_dim)
        
        