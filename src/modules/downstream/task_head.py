import torch
import torch.nn as nn
import torch.nn.functional as F

class TaskClassificationHead(nn.Module):
    def __init__(self, feature_dim, output_dim):
        super().__init__()

        self.fc1 = nn.Linear(feature_dim, output_dim)
    
    def forward(self, x):
        x = x.mean(dim=1)
        x = self.fc1(x)
        return x.squeeze(-1)