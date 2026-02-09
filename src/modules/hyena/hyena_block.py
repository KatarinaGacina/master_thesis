import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.transformer_encoder.ffn import FeedForward
from modules.hyena.hyenab_operator import HyenaOperator

import math


#function adjusted from https://github.com/Zymrael/savanna/tree/main
def get_exponential_decay(L, feature_dim, log_lambda_min=-1, log_lambda_max=2, shift=0.05, num_decay_repeats=1):
    assert feature_dim % num_decay_repeats == 0

    t = torch.abs(torch.linspace(-1, 1, L))[None] #bidirectional, odd kernel

    #later features -> faster_decay
    decay_rates = feature_dim // num_decay_repeats
    decay_range = torch.logspace(log_lambda_min, log_lambda_max, decay_rates)[:, None].repeat(num_decay_repeats, 1) 

    decay = torch.exp(-decay_range * t)

    return decay + shift

class HyenaOperatorMR(nn.Module):
    def __init__(self, feature_dim, inner_factor=1, order=2, kernel_size=3, filter_len=127, exp_decay=True):
        super().__init__()

        assert kernel_size % 2 == 1
        assert order >= 2

        self.feature_dim = feature_dim
        self.order = order

        self.in_proj = nn.Linear(feature_dim, (order + 1) * feature_dim * inner_factor)

        self.short_convs = nn.Conv1d(
            in_channels=(order + 1) * feature_dim * inner_factor,
            out_channels=(order + 1) * feature_dim * inner_factor,
            kernel_size=kernel_size,
            groups=(order + 1) * feature_dim * inner_factor,
            padding=(kernel_size - 1) // 2
        )

        self.long_filters = nn.ParameterList()
        for i in range(order - 1):
            filter_tensor = nn.Parameter(torch.randn(feature_dim * inner_factor, 1, filter_len))
            self.long_filters.append(filter_tensor)

        self.exp_decay = exp_decay
        if self.exp_decay:
            self.register_buffer("window", get_exponential_decay(filter_len, feature_dim).unsqueeze(1))
            #self.register_buffer("window", torch.hamming_window(filter_len, periodic=False)) #dummy, fixed local bias

        self.silu  = nn.SiLU()
        self.out_proj = nn.Linear(feature_dim * inner_factor, feature_dim * inner_factor)
    
    def forward(self, x):
        _, L, C = x.shape

        x = self.in_proj(x)

        x = x.transpose(1, 2)

        x = self.short_convs(x)
        short_conv_outputs = torch.chunk(x, self.order + 1, dim=1)

        out = short_conv_outputs[-1]
        for i, x_i in enumerate(reversed(short_conv_outputs[1:-1])):
            out = out * x_i

            long_filter = self.long_filters[i]
            if self.exp_decay:
                long_filter = long_filter * self.window

            out = F.conv1d(out, long_filter, padding="same", groups=C) #depthwise
        out = out * short_conv_outputs[0]

        out = out.transpose(1, 2)

        out = self.silu(out)
        out = self.out_proj(out)

        return out

class HyenaBlockMR(nn.Module):
    def __init__(self, feature_dim, filter_len, inner_factor=1, order=2, modulate=True):
        super().__init__()

        self.norm_in = nn.RMSNorm(feature_dim, eps=1e-8)
        self.hmr = HyenaOperatorMR(feature_dim, order=order, inner_factor=inner_factor, filter_len=filter_len, exp_decay=modulate)
        self.norm = nn.RMSNorm(feature_dim, eps=1e-8)
        self.ffn = FeedForward(feature_dim, 2*feature_dim)
        
    def forward(self, x, attn_mask=None):
        x_norm = self.norm_in(x)
        x_hmr = self.hmr(x_norm)
        x_hmr = x_hmr + x
        out = self.norm(x_hmr)
        out = self.ffn(out)
        out = out + x_hmr

        return out



class HyenaBlockLI(nn.Module):
    def __init__(self, feature_dim, l_max, inner_factor=1, order=2, is_causal=True, modulate=True):
        super().__init__()

        self.norm_in = nn.RMSNorm(feature_dim, eps=1e-8) 
        self.hmr = HyenaOperator(feature_dim, l_max, order=order, inner_factor=inner_factor, is_causal=is_causal, modulate=modulate)
        self.norm = nn.RMSNorm(feature_dim, eps=1e-8)
        self.ffn = FeedForward(feature_dim, 2*feature_dim)
        
    def forward(self, x, attn_mask=None):
        x_norm = self.norm_in(x)
        x_hmr = self.hmr(x_norm)
        x_hmr = x_hmr + x
        out = self.norm(x_hmr)
        out = self.ffn(out)
        out = out + x_hmr

        return out