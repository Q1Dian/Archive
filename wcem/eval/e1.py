"""E1 / Gate 1: recall NLL by lag for the trained model and write baselines."""
import torch

from wcem.eval.stage_a import BINS, memory_gain
from wcem.eval.triggers import writes_at_rate
from wcem.models.wcem import LAG, gaussian_nll

LAGS = (8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384)
FIFO_BYTES = 256 * 1024


def uniform_writes(n, rate):
    k = int(round(float(rate) * n))
    w = torch.zeros(n, dtype=torch.bool)
    if k <= 0:
        return w
    step = n / k
    idx = (torch.arange(k) * step + step - 1).long().clamp(0, n - 1)
    w[idx] = True
    return w


def fifo_keep_writes(n, rate, keep):
    """Uniform write schedule; only the last `keep` scheduled writes may open the gate."""
    w = uniform_writes(n, rate)
    idx = w.nonzero().flatten().tolist()[-keep:]
    g = torch.zeros(n, 2)
    for t in idx:
        g[t] = 1.0
    return g


def fifo_byte_cap_g(n, rate, heads=4, dk=128, dv=128, budget=FIFO_BYTES):
    """FIFO depth from the 256 KB episodic matrix size (one outer-product slot budget)."""
    keep = max(1, budget // (heads * dk * dv * 4))
    return fifo_keep_writes(n, rate, keep)


def fifo_index(tau, writes, keep):
    """Most recent write at or before tau that is still in the last-`keep` writes of the video."""
    idx = writes.nonzero().flatten().tolist()
    kept = idx[-keep:] if keep else []
    before = [i for i in kept if i <= tau]
    return before[-1] if before else -1


def nll_by_lag(nll, lags):
    out = {}
    for a, b, name in BINS:
        m = (lags >= a) & (lags < b)
        if m.any():
            out[name] = nll[m].mean().item()
    return out


def query_taus(n, window, lags=LAGS):
    taus, used = [], []
    last = n - 1
    for lag in lags:
        tau = last - lag
        if LAG <= tau <= last - window:
            taus.append(tau)
            used.append(lag)
    return taus, used


@torch.no_grad()
def seq_nll(model, out, z, S, taus):
    if not taus:
        return torch.zeros(z.shape[0], 0, device=z.device)
    tau = torch.tensor(taus, device=z.device)
    idx = tau[None].expand(z.shape[0], -1)
    x_q = out["x_lag"].gather(1, idx[..., None].expand(-1, -1, out["x_lag"].shape[-1])).float()
    z_q = z.gather(1, idx[..., None, None].expand(-1, -1, *z.shape[2:])).float()
    mu, ls = model.read_seq(S.float(), x_q)
    return gaussian_nll(z_q, mu, ls)


def gain_vs_swap(model, out, z, taus, used):
    S, S_swap = out["S_chunks"][-1], out["S_chunks"][-1].roll(1, 0)
    own = seq_nll(model, out, z, S, taus)
    swap = seq_nll(model, out, z, S_swap, taus)
    if own.numel() == 0 or z.shape[0] < 2:
        return {}, own, swap
    lag = torch.tensor(used, device=z.device).repeat(z.shape[0])
    return memory_gain(own.flatten(), swap.flatten(), lag), own, swap


def accumulate_nll_gain(own_parts, swap_parts, used, device):
    if not own_parts:
        return {}, 0
    own = torch.cat(own_parts)
    swap = torch.cat(swap_parts) if swap_parts else None
    lag = torch.tensor(used, device=device).repeat(own.shape[0] // len(used))
    gain = memory_gain(own.flatten(), swap.flatten(), lag) if swap is not None else {}
    return gain, own.numel() // len(used)
