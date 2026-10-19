"""Stage A training logs: memory gain by lag, write rate, written-key cosine, sigma clamp."""
import math

import torch

from wcem.models.wcem import LAG, gaussian_nll

BINS = ((8, 16, "8-16"), (16, 32, "16-32"), (32, 64, "32-64"), (64, 128, "64-128"),
        (128, 256, "128-256"), (256, math.inf, "256+"))


def writes_per_minute(writes_per_step):
    return float(writes_per_step) * 60.0


def lag_bin(lag):
    for a, b, name in BINS:
        if a <= lag < b:
            return name
    return "256+"


def memory_gain(nll_own, nll_swap, lags):
    """Mean (swap - own) NLL per lag bin. Positive means own memory helps."""
    gain = nll_swap - nll_own
    out = {}
    for a, b, name in BINS:
        m = (lags >= a) & (lags < b)
        if m.any():
            out[name] = gain[m].mean().item()
    return out


def written_key_cosine(k, p_open, thresh=0.5):
    """Mean cosine among keys of a step whose pair opened (over batch and heads)."""
    k = torch.nn.functional.normalize(k, dim=-1)
    w = p_open >= thresh
    if k.shape[2] >= 2:
        pair = (k[:, :, 0] * k[:, :, 1]).sum(-1)
        mask = w[:, :, 0] if w.dim() == 3 else w
        if mask.any():
            return pair[mask].mean().item()
    return float("nan")


def sigma_clamp_frac(log_sigma, rng):
    lo, hi = rng
    x = torch.as_tensor(log_sigma)
    return ((x <= lo) | (x >= hi)).float().mean().item()


@torch.no_grad()
def recall_gain(model, out, z, lags=(8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384)):
    """Own vs batch-swapped S, queried from the last chunk state."""
    B, N = z.shape[:2]
    n = N - 1
    S, S_swap = out["S_chunks"][-1], out["S_chunks"][-1].roll(1, 0)
    taus, used = [], []
    for lag in lags:
        tau = n - lag
        if LAG <= tau <= n - model.cfg.window:
            taus.append(tau)
            used.append(lag)
    if not taus or B < 2:
        return {}, float("nan")
    tau = torch.tensor(taus, device=z.device)
    idx = tau[None].expand(B, -1)
    x_q = out["x_lag"].gather(1, idx[..., None].expand(-1, -1, out["x_lag"].shape[-1])).float()
    z_q = z.gather(1, idx[..., None, None].expand(-1, -1, *z.shape[2:])).float()
    mu, ls = model.read_seq(S, x_q)
    own = gaussian_nll(z_q, mu, ls)
    mu_s, ls_s = model.read_seq(S_swap, x_q)
    swap = gaussian_nll(z_q, mu_s, ls_s)
    lag = torch.tensor(used, device=z.device).repeat(B)
    return memory_gain(own.flatten(), swap.flatten(), lag), sigma_clamp_frac(ls, model.cfg.log_sigma_range)
