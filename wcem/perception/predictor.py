"""Stage 0 development predictor (canvas part P).

Given encoder tokens of the context [t, t+2) s (temporal tokens 0-3), predict
the full-clip encoder tokens of step n (temporal tokens 4-5). Per-patch L1
between prediction and target is the Stage 0 error e_n, kept as an M8
baseline trigger. The model's gate input is the token feature difference
(`wcem.data.surprise.token_diffs`), not e_n.
"""
import torch
import torch.nn as nn

from wcem.models.blocks import SelfBlock

T_CTX, T_TGT, HW = 4, 2, 24


class Predictor(nn.Module):
    def __init__(self, in_dim=1024, dim=384, depth=6, heads=6):
        super().__init__()
        self.inp = nn.Linear(in_dim, dim)
        self.pos_t = nn.Parameter(torch.randn(T_CTX + T_TGT, 1, 1, dim) * 0.02)
        self.pos_h = nn.Parameter(torch.randn(1, HW, 1, dim) * 0.02)
        self.pos_w = nn.Parameter(torch.randn(1, 1, HW, dim) * 0.02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.blocks = nn.ModuleList(SelfBlock(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, in_dim)

    def pos(self):
        return (self.pos_t + self.pos_h + self.pos_w).flatten(0, 2)

    def forward(self, ctx):
        """ctx [B, 4 * 576, D] -> prediction [B, 2 * 576, D]."""
        B = ctx.shape[0]
        pos = self.pos()
        n_ctx = T_CTX * HW * HW
        h = torch.cat([self.inp(ctx) + pos[:n_ctx], self.mask_token.expand(B, T_TGT * HW * HW, -1) + pos[n_ctx:]], 1)
        for blk in self.blocks:
            h = blk(h)
        return self.out(self.norm(h[:, n_ctx:]))


def error_map(pred, target):
    """Per-patch L1, [B, 2 * 576] -> [B, 2, 24, 24]."""
    return (pred.float() - target.float()).abs().mean(-1).view(-1, T_TGT, HW, HW)
