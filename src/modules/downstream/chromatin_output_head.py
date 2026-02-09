import torch
import torch.nn as nn
import torch.nn.functional as F


class ChromatinClassificationHead(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()

        self.fc1 = nn.Linear(feature_dim, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight) 
        nn.init.zeros_(self.fc1.bias)  #bias always 0 due to normalization in previous block
    
    def forward(self, x):
        x = self.fc1(x)
        return x.squeeze(-1)