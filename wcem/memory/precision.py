"""Precision-weighted fusion of slow L and fast S predictions (Tier 1)."""
import torch

EPS = 1e-6


def fuse_explained(v, mu_L, var_L, mu_S, sigma_S=1.0):
    """Combine mu_L and mu_S by inverse-variance weights; return residual v' = v - v_hat.

    mu_L, mu_S, v: [..., H, dv]. var_L: [..., H] epistemic variance from ensemble spread.
    """
    inv_l = 1.0 / (var_L[..., None] + EPS)
    inv_s = 1.0 / (float(sigma_S) ** 2 + EPS)
    w_l = inv_l / (inv_l + inv_s)
    v_hat = w_l * mu_L + (1.0 - w_l) * mu_S
    return v - v_hat
