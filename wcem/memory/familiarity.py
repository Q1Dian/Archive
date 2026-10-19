"""Diagonal key-energy familiarity for Kalman-style write gain (Tier 2)."""
import torch

EPS = 1e-6


class FamiliarityState:
    """Tracks c_h in R^dk per head: familiarity(k) = sum_d k_d^2 * c_d."""

    def __init__(self, batch, heads, dk, device, dtype):
        self.c = torch.zeros(batch, heads, dk, device=device, dtype=dtype)

    def score(self, k):
        """k [B, H, dk] -> [B, H] familiarity (higher = more prior writes on this direction)."""
        return (k.square() * self.c).sum(-1)

    def update(self, k, g, alpha=0.9998):
        """g [B, pairs] or [B, H] gate mass; accumulate keyed energy."""
        if g.dim() == 2 and g.shape[1] * 2 == k.shape[1]:
            g = g.repeat_interleave(2, dim=1)
        mass = g[..., None]
        self.c = alpha * self.c + mass * k.square()


def kalman_gain(familiarity, sigma_obs=1.0, sigma_prior=1.0):
    """beta in (0,1]: high when unfamiliar, low when familiar."""
    sigma_mem = sigma_prior / (familiarity + 1.0)
    return (sigma_mem / (sigma_mem + float(sigma_obs) + EPS)).clamp(0.0, 1.0)
