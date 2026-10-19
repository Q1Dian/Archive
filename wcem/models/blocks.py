import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim, hidden, out=None):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(dim, hidden), nn.Linear(hidden, out or dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class CrossBlock(nn.Module):
    """Pre-norm cross-attention (queries attend to context) followed by an MLP."""

    def __init__(self, dim, heads, ctx_dim=None):
        super().__init__()
        ctx_dim = ctx_dim or dim
        self.nq, self.nc = nn.LayerNorm(dim), nn.LayerNorm(ctx_dim)
        self.attn = nn.MultiheadAttention(dim, heads, kdim=ctx_dim, vdim=ctx_dim, batch_first=True)
        self.nm, self.mlp = nn.LayerNorm(dim), MLP(dim, 4 * dim)

    def forward(self, q, ctx, key_padding_mask=None):
        c = self.nc(ctx)
        q = q + self.attn(self.nq(q), c, c, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return q + self.mlp(self.nm(q))


class SelfBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.n1, self.attn = nn.LayerNorm(dim), nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2, self.mlp = nn.LayerNorm(dim), MLP(dim, 4 * dim)

    def forward(self, x, key_padding_mask=None):
        h = self.n1(x)
        x = x + self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class AttnPool(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.block = CrossBlock(dim, heads)

    def forward(self, x, key_padding_mask=None):
        q = self.q.expand(x.shape[0], -1, -1)
        return self.block(q, x, key_padding_mask)[:, 0]
