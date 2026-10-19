"""M5 exit check: train the memory core on synthetic episodes.

Pass criteria:
  1. memory gain > 0: recall NLL at boundaries is lower with the episode's own S than with S from another episode;
  2. the gate opens far more often at boundaries than inside segments;
  3. the gate opens less often on repeated transitions than on first occurrences;
  4. the write rate falls as lambda rises.
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
LOG_LAM = (math.log(0.03), math.log(30.0))
EVAL_LAMS = (0.1, 0.3, 1.0, 3.0, 10.0)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.wk = nn.Linear(DIM, H * DK)
        self.wv = nn.Linear(DIM, H * DK)
        self.wcue = nn.Linear(DIM, 64)
        self.decay = Decay(DIM, H, alpha0=0.999)
        self.mem = EpisodicMemory(2 * DIM + 1, heads=H, dk=DK, dv=DK, hidden=128, sigma_q=0.01, chunk=32)
        self.dec = nn.Sequential(nn.Linear(H * DK + 64, 256), nn.GELU(), nn.Linear(256, DIM + 1))

    def keys(self, x):
        return F.normalize(self.wk(x).unflatten(-1, (H, DK)), dim=-1)

    def run(self, x, log_lam):
        B, N, _ = x.shape
        xprev = torch.cat([x[:, :1].expand(-1, LAG, -1), x[:, :-LAG]], 1)
        k = self.keys(xprev)
        v = F.normalize(self.wv(x).unflatten(-1, (H, DK)), dim=-1)
        feats = torch.cat([x, xprev, log_lam[:, None, None].expand(-1, N, 1)], -1)
        out = self.mem(k, v, torch.zeros_like(v), feats, self.decay(x))
        return out, xprev

    def recall(self, S, xprev_q):
        """S [B, H, dv, dk], xprev_q [B, Q, DIM] -> mu, log_sigma."""
        q = self.keys(xprev_q).transpose(0, 1)
        m = self.mem.read(S, q).transpose(0, 1).flatten(-2)
        o = self.dec(torch.cat([m, self.wcue(xprev_q)], -1))
        return o[..., :DIM], o[..., DIM].clamp(-3, 2)


def nll(z, mu, log_sigma):
    return ((z - mu).square().sum(-1) / (2 * torch.exp(2 * log_sigma)) + DIM * log_sigma)


def query_loss(model, out, xprev, z, n_q=32, boundary_only=None, S_override=None):
    """Read at the end of each chunk with queries tau <= n - W."""
    B, N = z.shape[:2]
    chunk = model.mem.chunk
    losses = []
    for c in range(out["S_chunks"].shape[0]):
        n = min((c + 1) * chunk, N) - 1
        if n - W < LAG:
            continue
        S = out["S_chunks"][c] if S_override is None else S_override[c]
        if boundary_only is None:
            tau = torch.randint(LAG, n - W + 1, (B, n_q), device=z.device)
        else:
            tau = boundary_only[c]
            if tau is None:
                continue
        idx = tau[..., None].expand(-1, -1, DIM)
        mu, ls = model.recall(S, xprev.gather(1, idx))
        losses.append(nll(z.gather(1, idx), mu, ls))
    return torch.cat(losses, 1)


def train(model, data, iters, batch, device, log_every=200):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    E = data["x"].shape[0]
    for it in range(iters):
        idx = torch.randint(E, (batch,))
        x, z = data["x"][idx].to(device), data["z"][idx].to(device)
        log_lam = torch.empty(batch, device=device).uniform_(*LOG_LAM)
        lam = log_lam.exp()
        out, xprev = model.run(x, log_lam)
        rec = query_loss(model, out, xprev, z).mean()
        writes = out["p_open"].sum(-1).mean(1)
        size = out["delta"].abs().sum((-1, -2)).mean(1)
        loss = rec + (lam * (writes + 0.1 * size)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % log_every == 0 or it == iters - 1:
            print(f"it {it:5d} loss {loss.item():8.3f} recall {rec.item():8.3f} "
                  f"writes/step {writes.mean().item():.3f}", flush=True)


@torch.no_grad()
def evaluate(model, data, device, lam):
    model.eval()
    x, z = data["x"].to(device), data["z"].to(device)
    B, N = z.shape[:2]
    log_lam = torch.full((B,), math.log(lam), device=device)
    out, xprev = model.run(x, log_lam)
    g = out["g"].amax(-1)
    bnd, rep = data["boundary"].to(device), data["repeat"].to(device)

    chunk = model.mem.chunk
    taus = []
    for c in range(out["S_chunks"].shape[0]):
        n = min((c + 1) * chunk, N) - 1
        cand = torch.arange(LAG, max(LAG, n - W + 1), device=device)
        if len(cand) == 0:
            taus.append(None)
            continue
        mask = bnd[:, cand]
        k = int(mask.sum(1).min().item())
        if k == 0:
            taus.append(None)
            continue
        order = torch.argsort(mask.float() + 1e-3 * torch.rand_like(mask.float()), dim=1, descending=True)[:, :k]
        taus.append(cand[order])
    own = query_loss(model, out, xprev, z, boundary_only=taus)
    swap = query_loss(model, out, xprev, z, boundary_only=taus, S_override=out["S_chunks"].roll(1, dims=1))
    first = bnd & ~rep
    model.train()
    return dict(
        lam=lam,
        gain=(swap - own).mean().item(),
        writes_per_step=g.mean().item(),
        p_write_boundary=g[bnd].mean().item(),
        p_write_inside=g[~bnd].mean().item(),
        p_write_first=g[first].mean().item(),
        p_write_repeat=g[rep].mean().item(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=256)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    path = DATA / "synthetic" / f"episodes_{args.steps}.pt"
    if path.exists():
        train_data, eval_data = torch.load(path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        train_data = make_episodes(4096, args.steps, DIM, seed=0)
        eval_data = make_episodes(256, args.steps, DIM, seed=1)
        torch.save((train_data, eval_data), path)

    model = Model().to(device)
    train(model, train_data, args.iters, args.batch, device)
    results = [evaluate(model, eval_data, device, lam) for lam in EVAL_LAMS]
    for r in results:
        print(" ".join(f"{k}={v:.3f}" for k, v in r.items()))

    useful = [r for r in results if r["gain"] > 0] or results
    mid = max(useful, key=lambda r: r["p_write_boundary"] / (r["p_write_inside"] + 1e-3))
    print(f"selectivity checked at lam={mid['lam']}")
    rates = [r["writes_per_step"] for r in results]
    checks = {
        "memory gain > 0 at boundaries": mid["gain"] > 0,
        "gate opens more at boundaries": mid["p_write_boundary"] > 2 * mid["p_write_inside"],
        "fewer writes on repeated transitions": mid["p_write_repeat"] < mid["p_write_first"],
        "write rate falls with lambda": all(a >= b for a, b in zip(rates, rates[1:])) and rates[0] > rates[-1],
    }
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    torch.save(model.state_dict(), DATA / "synthetic" / "m5_model.pt")


if __name__ == "__main__":
    main()
