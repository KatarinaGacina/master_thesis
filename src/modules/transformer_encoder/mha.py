import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from modules.positional_embed.rope import RotaryPositionalEmbeddings

#from torch.nn.attention.flex_attention import flex_attention
#from torch.nn.attention.flex_attention import create_block_mask, and_masks
#from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func, flash_attn_varlen_func
#from flash_attn.bert_padding import unpad_input, pad_input


class MultiHeadAttn(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads

        self.rope_pos = RotaryPositionalEmbeddings(self.head_dim)

        #self.q_proj = nn.Linear(embed_dim, embed_dim)
        #self.k_proj = nn.Linear(embed_dim, embed_dim)
        #self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-8) #nn.LayerNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-8) #nn.LayerNorm(self.head_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_uniform_(self.qkv_proj.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.out_proj.weight, mode='fan_in', nonlinearity='relu') 

        #nn.init.ones_(self.q_norm.weight)
        #nn.init.zeros_(self.q_norm.bias)
        #nn.init.ones_(self.k_norm.weight) 
        #nn.init.zeros_(self.k_norm.bias)
    
    def forward(self, x, attn_mask=None):
        B, L, _ = x.size()
        H = self.num_heads
        h_dim = self.head_dim
        
        qkv = self.qkv_proj(x)

        qkv = qkv.reshape(B, L, H, 3*h_dim)
        qkv = qkv.transpose(1, 2)
        q, k, v = torch.split(qkv, qkv.size(3) // 3, dim=3) #same dimensions

        q = self.q_norm(q) #normalization per head
        k = self.k_norm(k)

        q = self.rope_pos.forward(q)
        k = self.rope_pos.forward(k)

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn_out = attn_out.transpose(1, 2).flatten(start_dim=2)

        out = self.out_proj(attn_out)

        return out
