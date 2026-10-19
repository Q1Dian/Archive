"""Causal per-patch normalization of the step signals, and the per-step summary.

Gate 0 (EK100 held out): the per-cell 0.5 s token feature difference ties the
Stage 0 error on action-boundary F1 and beats it on AUC, so the gate's surprise
input is the normalized token feature difference (token_diffs, diff_zscore).
The Stage 0 error stays as a baseline trigger.

Errors arrive as [N, 2, 576]: step n, its two temporal tokens (0.5 s each), 24x24 patches.
Tokens are processed in time order. Each token is z-scored with the running mean and
variance from strictly earlier tokens, then the statistics are updated, so z at time t
never sees t. Statistics start from a prior (per-patch train-set mean and variance) and
decay with time constant TAU seconds.
"""
import math

import torch

TAU = 60.0
TOKEN_SECONDS = 0.5
EPS = 1e-6


def ema_zscore(err, prior_mean, prior_var, tau=TAU, dt=TOKEN_SECONDS):
    """err [N, 2, P] -> z [N, 2, P] float32. prior_* [P]."""
    x = torch.as_tensor(err, dtype=torch.float32).flatten(0, 1)
    a = 1.0 - math.exp(-dt / tau)
    m = torch.as_tensor(prior_mean, dtype=torch.float32).clone()
    v = torch.as_tensor(prior_var, dtype=torch.float32).clone()
    z = torch.empty_like(x)
    for t in range(len(x)):
        d = x[t] - m
        z[t] = d / (v + EPS).sqrt()
        m += a * d
        v = (1 - a) * (v + a * d * d)
    return z.view_as(torch.as_tensor(err))


def token_diffs(tcells):
    """Per-cell feature change into each target token: tcells [N, 2, C, D] -> [N, 2, C].

    [:, 1] compares the two tokens of a step (same clip); [:, 0] compares token 0
    of step n with token 1 of step n - 1 (adjacent clips; step 0 repeats [:, 1]).
    """
    c = torch.nn.functional.normalize(torch.as_tensor(tcells, dtype=torch.float32), dim=-1)
    d = torch.empty(c.shape[:3])
    d[:, 1] = (c[:, 1] - c[:, 0]).norm(dim=-1)
    d[1:, 0] = (c[1:, 0] - c[:-1, 1]).norm(dim=-1)
    d[0, 0] = d[0, 1]
    return d


def diff_prior(diffs):
    """Per-cell mean and variance per stream (across-clip, within-clip) from a list of [N, 2, C]."""
    return [prior_from([d[:, k:k + 1] for d in diffs]) for k in (0, 1)]


def diff_zscore(d, prior, tau=TAU):
    """Causal z-score of token_diffs output, each stream with its own statistics at 1 s."""
    return torch.cat([ema_zscore(d[:, k:k + 1], *prior[k], tau=tau, dt=1.0) for k in (0, 1)], 1)


def summary(z):
    """z [N, 2, P] -> [N, 4]: mean, 90th percentile, max, fraction above 2 sigma."""
    f = z.flatten(1)
    return torch.stack([f.mean(1), f.quantile(0.9, dim=1), f.amax(1), (f > 2).float().mean(1)], 1)


def prior_from(errs):
    """Per-patch mean and variance over a list of [N, 2, P] arrays (train videos)."""
    x = torch.cat([torch.as_tensor(e, dtype=torch.float32).flatten(0, 1) for e in errs])
    return x.mean(0), x.var(0)
