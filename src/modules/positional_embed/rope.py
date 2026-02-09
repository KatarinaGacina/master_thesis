import torch
import torch.nn as nn
import torch.nn.functional as F

class RotaryPositionalEmbeddings(nn.Module):

    def __init__(self, d, base=10000):
        super().__init__()

        assert d % 2 == 0, "2D slices"

        self.base = base
        self.d = d

        self.cos_precompute = None
        self.sin_precompute = None

    def _build_precompute(self, x):
        _, _, seq_len, _ = x.shape
        device = x.device

        if self.cos_precompute is not None and seq_len <= self.cos_precompute.shape[0]:
            return

        theta = 1.0 / (self.base ** (torch.arange(0, self.d, 2).float() / self.d)).to(device)
        pos_idx = torch.arange(seq_len, device=device, dtype=torch.float)

        idx_theta = torch.einsum('n,d->nd', pos_idx, theta)
        idx_theta = torch.cat([idx_theta, idx_theta], dim=1)

        self.cos_precompute = torch.cos(idx_theta)[None, None, :, :] #elementwise None -> 1
        self.sin_precompute = torch.sin(idx_theta)[None, None, :, :]

    def forward(self, x):
        self._build_precompute(x)

        _, _, seq_len, _ = x.shape

        #x_rope, x_pass = x[..., :self.d], x[..., self.d:] #u ovom slucaju koristimo sve dimenzije pa ne treba

        x_1 = x[..., : (self.d // 2)]
        x_2 = x[..., (self.d // 2) :]

        neg_x = torch.cat([-x_2, x_1], dim=-1)

        x_rope = (x * self.cos_precompute[:, :, :seq_len, :]) + (neg_x * self.sin_precompute[:, :, :seq_len, :]) #needed if max_seq_len precomputed is longer than seq_len

        return x_rope


if __name__ == "__main__":
    rope = RotaryPositionalEmbeddings(2)

    q = torch.randn(2, 2, 4, 2)

    q_rope = rope(q)
    print("Original shape:", q.shape)
    print("After RoPE shape:", q_rope.shape)

    print("Original q[0,0]:", q[0, 0])
    print("RoPE q[0,0]:", q_rope[0, 0])

    print("Original norms:", q.norm(dim=-1))
    print("RoPE norms:", q_rope.norm(dim=-1))

    k = q.clone()
    k_rope = rope(k)
    attn_before = torch.matmul(q, k.transpose(-2, -1))
    attn_after = torch.matmul(q_rope, k_rope.transpose(-2, -1))
    print("Attention before RoPE:\n", attn_before[0,0])
    print("Attention after RoPE:\n", attn_after[0,0])

    q = torch.tensor([[[[1.0, 0.0],
                    [0.0, 1.0]]]]) 
    print(rope(q))
