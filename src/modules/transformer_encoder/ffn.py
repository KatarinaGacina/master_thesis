import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()

        self.lin1 = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.silu1  = nn.SiLU()
        self.lin2 = nn.Linear(hidden_dim, feature_dim, bias=False)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_uniform_(self.lin1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.lin2.weight, mode='fan_in', nonlinearity='relu')
    
    def forward(self, x):
        out = self.lin1(x)
        out = self.silu1(out)
        out = self.lin2(out) 
        return out
