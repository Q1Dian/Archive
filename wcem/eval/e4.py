"""E4: write rate on near-duplicate (repeat) steps vs first visits (Gate 2)."""
import numpy as np
import torch

SIM_THRESH = 0.85


@torch.no_grad()
def repeat_write_stats(model, tokens, z, surprise, dupsim, log_lam, sim_thresh=SIM_THRESH):
    """dupsim [N, K] cosine to nearest non-local neighbors from precompute."""
    device = z.device
    lam = torch.full((1,), float(log_lam), device=device)
    t = tokens[None].to(device)
    if t.dtype == torch.bfloat16:
        t = t.float()
    out = model(t, z[None].float(), surprise[None].float(), lam)
    p_open = out["p_open"][0].mean(-1).cpu().numpy()
    sim = np.asarray(dupsim, dtype=np.float32)
    n = min(len(p_open), len(sim))
    first, repeat = [], []
    for t in range(2, n):
        if sim[t, 0] >= sim_thresh:
            repeat.append(float(p_open[t]))
        else:
            first.append(float(p_open[t]))
    pf = float(np.mean(first)) if first else float("nan")
    pr = float(np.mean(repeat)) if repeat else float("nan")
    p_seq = p_open[:n]
    return dict(
        p_open_mean=float(np.mean(p_seq)),
        p_open_std=float(np.std(p_seq)),
        p_write_first=pf,
        p_write_repeat=pr,
        n_first=len(first),
        n_repeat=len(repeat),
        repeat_minus_first=(pr - pf) if first and repeat else float("nan"),
        repeat_over_first=(pr / pf) if first and repeat and pf > 0 else float("nan"),
        repeat_lt_first=bool(pr < pf) if first and repeat else None,
    )
