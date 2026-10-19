"""M5.5: Kalman-style familiarity gate on synthetic episodes (no PairGate MLP).

Pass if at mid-lambda: memory gain > 0 and p_write_repeat < p_write_first.
"""
import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wcem.memory.episodic import EpisodicMemory  # noqa: E402
from wcem.memory.gates import Decay  # noqa: E402
from wcem.paths import DATA  # noqa: E402
from wcem.synthetic import make_episodes  # noqa: E402

DIM, H, DK, W, LAG = 32, 2, 32, 4, 2
EVAL_LAM = 3.0


class KalmanModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.wk = nn.Linear(DIM, H * DK)
        self.wv = nn.Linear(DIM, H * DK)
        self.wcue = nn.Linear(DIM, 64)
        self.decay = Decay(DIM, H, alpha0=0.999)
        self.mem = EpisodicMemory(2 * DIM + 1, heads=H, dk=DK, dv=DK, hidden=128, sigma_q=0.0, chunk=32)

    def keys(self, x):
        return F.normalize(self.wk(x).unflatten(-1, (H, DK)), dim=-1)

    def run(self, x, log_lam):
        B, N, _ = x.shape
        xprev = torch.cat([x[:, :1].expand(-1, LAG, -1), x[:, :-LAG]], 1)
        k = self.keys(xprev)
        v = F.normalize(self.wv(x).unflatten(-1, (H, DK)), dim=-1)
        feats = torch.cat([x, xprev, log_lam[:, None, None].expand(-1, N, 1)], -1)
        out = self.mem(k, v, torch.zeros_like(v), feats, self.decay(x), kalman_gate=True, sigma_obs=0.5)
        return out, xprev


def nll(z, mu, log_sigma):
    return ((z - mu).square().sum(-1) / (2 * torch.exp(2 * log_sigma)) + DIM * log_sigma)


@torch.no_grad()
def evaluate(model, data, device):
    model.eval()
    x, z = data["x"].to(device), data["z"].to(device)
    B = x.shape[0]
    log_lam = torch.full((B,), math.log(EVAL_LAM), device=device)
    out, _ = model.run(x, log_lam)
    g = out["g"].amax(-1)
    bnd, rep = data["boundary"].to(device), data["repeat"].to(device)
    first = bnd & ~rep
    return dict(
        p_write_first=g[first].mean().item() if first.any() else float("nan"),
        p_write_repeat=g[rep].mean().item() if rep.any() else float("nan"),
        writes_per_step=g.mean().item(),
    )


def train(model, data, iters, batch, device):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    E = data["x"].shape[0]
    for it in range(iters):
        idx = torch.randint(E, (batch,))
        x = data["x"][idx].to(device)
        log_lam = torch.full((batch,), math.log(EVAL_LAM), device=device)
        out, _ = model.run(x, log_lam)
        loss = out["delta"].square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--steps", type=int, default=256)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = DATA / "synthetic" / f"episodes_{args.steps}.pt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        train_data = make_episodes(4096, args.steps, DIM, seed=0)
        eval_data = make_episodes(256, args.steps, DIM, seed=1)
        torch.save((train_data, eval_data), path)
    else:
        train_data, eval_data = torch.load(path, weights_only=False)

    model = KalmanModel().to(device)
    train(model, train_data, args.iters, 32, device)
    r = evaluate(model, eval_data, device)
    ok = r["p_write_repeat"] < r["p_write_first"]
    print(r, flush=True)
    print(f"{'PASS' if ok else 'FAIL'}  fewer writes on repeats (M5.5)")
    out = DATA / "synthetic" / "m5_kalman_result.json"
    import json
    out.write_text(json.dumps(dict(passed=ok, **r), indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
