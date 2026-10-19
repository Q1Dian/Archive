"""Slow memory L: prototype library ensemble shared across videos (canvas section 3.6).

L_e^h(k) = sum_j softmax_j(k . P_{e,j}^h / tau) U_{e,j}^h, with unit-norm U.
Stage A uses `enabled=False`, where L returns zeros.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SlowMemory(nn.Module):
    def __init__(self, heads=4, dk=128, dv=128, prototypes=1024, members=5, tau=0.1, enabled=False):
        super().__init__()
        self.enabled, self.tau = enabled, tau
        self.P = nn.Parameter(F.normalize(torch.randn(members, heads, prototypes, dk), dim=-1))
        self.U = nn.Parameter(torch.randn(members, heads, prototypes, dv) * 0.02)

    def members(self, k):
        """k [..., H, dk] -> per-member predictions [E, ..., H, dv] and assignments [E, ..., H, M]."""
        logits = torch.einsum("...hk,ehmk->e...hm", k, F.normalize(self.P, dim=-1)) / self.tau
        a = logits.softmax(-1)
        return torch.einsum("e...hm,ehmv->e...hv", a, F.normalize(self.U, dim=-1)), a

    def forward(self, k):
        if not self.enabled:
            return torch.zeros_like(k[..., : self.U.shape[-1]])
        return self.members(k)[0].mean(0)

    def forward_with_variance(self, k):
        """Mean prediction and per-head epistemic variance (ensemble spread)."""
        if not self.enabled:
            z = torch.zeros_like(k[..., : self.U.shape[-1]])
            return z, torch.ones(k.shape[:-1], device=k.device, dtype=k.dtype)
        preds, _ = self.members(k)
        mu = preds.mean(0)
        var = preds.var(0, unbiased=False).mean(-1).clamp_min(1e-4)
        return mu, var
