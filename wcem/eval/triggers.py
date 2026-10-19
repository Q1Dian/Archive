"""M8 write-trigger baselines, read from the precomputed `_trig.npy` columns.

Each trigger keeps the top-k local maxima so the write rate matches a trained
run (or any other target rate). Columns: featdiff, err_raw, err_norm, fd_raw,
fd_norm (Gate 0 signal).
"""
import torch


def writes_at_rate(signal, rate):
    """Boolean writes [N]: local maxima, then the top `round(rate * N)` by height."""
    s = torch.as_tensor(signal, dtype=torch.float32).flatten()
    n = len(s)
    k = int(round(float(rate) * n))
    out = torch.zeros(n, dtype=torch.bool)
    if k <= 0 or n == 0:
        return out
    p = torch.nn.functional.pad(s[None, None], (1, 1), value=-float("inf"))[0, 0]
    peaks = (s >= p[:-2]) & (s > p[2:])
    idx = peaks.nonzero().flatten()
    if len(idx) == 0:
        idx = s.topk(min(k, n)).indices
    else:
        idx = idx[s[idx].topk(min(k, len(idx))).indices]
    out[idx[:k]] = True
    return out
