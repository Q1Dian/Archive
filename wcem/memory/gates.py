"""Hard-concrete gate (Louizos et al., 2018) and the learned decay."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

BETA, GAMMA, ZETA = 2.0 / 3.0, -0.1, 1.1
LOG_RATIO = math.log(-GAMMA / ZETA)


def hard_concrete(log_alpha, training):
    """Sample z in [0, 1], exactly 0 or 1 with positive probability. At eval, z is 0 or 1."""
    if not training:
        return (log_alpha > 0).to(log_alpha.dtype)
    u = torch.rand_like(log_alpha).clamp(1e-6, 1 - 1e-6)
    s = torch.sigmoid((torch.log(u) - torch.log1p(-u) + log_alpha) / BETA)
    return (s * (ZETA - GAMMA) + GAMMA).clamp(0.0, 1.0)


def prob_open(log_alpha):
    """P(z != 0), the differentiable expected gate count."""
    return torch.sigmoid(log_alpha - BETA * LOG_RATIO)


class Decay(nn.Module):
    """alpha_n = exp(-softplus(a) * softplus(w . x_n + b)) per head, initialised near alpha0."""

    def __init__(self, x_dim, heads, alpha0=0.9998):
        super().__init__()
        self.proj = nn.Linear(x_dim, heads)
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0.0)
        target = -math.log(alpha0) / F.softplus(torch.tensor(0.0)).item()
        self.a = nn.Parameter(torch.full((heads,), math.log(math.expm1(target))))

    def forward(self, x):
        return torch.exp(-F.softplus(self.a) * F.softplus(self.proj(x)))
