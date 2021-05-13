import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class encodeToTraj(nn.Module):
    def __init__(self, latent_size):
        super(encodeToTraj, self).__init__()

        self.l1 = nn.Linear(latent_size, 64)
        self.l2 = nn.Linear(64, 512 )
        self.l3 = nn.Linear(512, 1024)
        self.l4 = nn.Linear(1024, 256)
        self.l5 = nn.Linear(256, 68)
    
    def forward(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))
        h = F.relu(self.l4(h))
        h = torch.sigmoid(self.l5(h))

        return h