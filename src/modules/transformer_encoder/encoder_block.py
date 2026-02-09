import torch.nn as nn
import torch.nn.functional as F

from modules.transformer_encoder.mha import MultiHeadAttn
from modules.transformer_encoder.ffn import FeedForward


class EncoderBlock(nn.Module):
    def __init__(self, feature_dim, num_heads, dropout_rate=0.3):
        super().__init__()

        self.norm_in = nn.RMSNorm(feature_dim, eps=1e-8)
        self.mha = MultiHeadAttn(feature_dim, num_heads)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm = nn.RMSNorm(feature_dim, eps=1e-8)
        self.ffn = FeedForward(feature_dim, 2*feature_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        
    def forward(self, x, attn_mask=None):
        x_norm = self.norm_in(x)
        x_mha = self.mha(x_norm, attn_mask)
        x_mha = self.dropout1(x_mha)
        x_mha = x_mha + x
        out = self.norm(x_mha)
        out = self.ffn(out)
        out = self.dropout2(out)
        out = out + x_mha

        return out


class EncoderN(nn.Module):
    def __init__(self, n, num_heads, feature_dim):
        super().__init__()

        self.num_heads = num_heads
        self.feature_dim = feature_dim

        assert n > 0
        self.n = n

        self.model_list = nn.ModuleList(
            [EncoderBlock(self.feature_dim, self.num_heads) for _ in range(self.n)]
        )
        self.norm = nn.RMSNorm(self.feature_dim, eps=1e-8)


    def forward(self, x, attn_mask=None):
        out = x

        for block in self.model_list:
            out = block(out, attn_mask=attn_mask)

        out = self.norm(out)
        return out