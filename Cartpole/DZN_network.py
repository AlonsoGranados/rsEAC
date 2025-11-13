import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DZN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DZN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
        # return torch.exp(self.layer3(x))
        # return torch.sqrt(0.00001 + torch.abs(self.layer3(x)))

class stable_DZN(nn.Module):
    def __init__(self, n_observations, n_actions, hidden_dim = 128):
        super(stable_DZN, self).__init__()
        # Q1 architecture
        self.linear1 = nn.Linear(n_observations, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, n_actions)


    def forward(self, x):
        x1 = F.relu(self.linear1(x))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)
        return x1


