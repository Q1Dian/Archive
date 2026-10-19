"""M3: train the Stage 0 development predictor on the token cache, then evaluate.

    python scripts/stage0_train.py train --seed 0
    python scripts/stage0_train.py train --seed 1
    python scripts/stage0_train.py eval            # error maps + two-seed agreement

Exit check (canvas): held-out L1 below the copy-last and mean baselines, and
per-clip Spearman agreement of the two seeds' error maps > 0.8.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wcem.paths import CACHE, CHECKPOINTS, MANIFESTS  # noqa: E402
from wcem.perception.predictor import Predictor, error_map  # noqa: E402

SRC = CACHE / "stage0"
OUT = CHECKPOINTS / "stage0"


def bf16(a):
    return torch.from_numpy(np.ascontiguousarray(a)).view(torch.bfloat16)


class TokenCache(Dataset):
    def __init__(self, split):
        vids = (MANIFESTS / f"ek100_d2_{split}.txt").read_text().split()
        self.vids = [v for v in vids if (SRC / f"{v}.json").exists()]
        self.ctx = {v: np.load(SRC / f"{v}_ctx.npy", mmap_mode="r") for v in self.vids}
        self.tgt = {v: np.load(SRC / f"{v}_tgt.npy", mmap_mode="r") for v in self.vids}
        self.index = [(v, i) for v in self.vids for i in range(len(self.ctx[v]))]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, j):
        v, i = self.index[j]
        return bf16(self.ctx[v][i]), bf16(self.tgt[v][i])


def train(args):
    torch.manual_seed(args.seed)
    data = TokenCache("train")
    print(f"train: {len(data.vids)} videos, {len(data)} clips", flush=True)
    loader = DataLoader(data, args.batch, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True,
                        persistent_workers=True)
    model = Predictor().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05, betas=(0.9, 0.95))
    total = args.epochs * len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    OUT.mkdir(parents=True, exist_ok=True)
    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        for ctx, tgt in loader:
            ctx, tgt = ctx.cuda(non_blocking=True), tgt.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(ctx)
            loss = (pred.float() - tgt.float()).abs().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 100 == 0:
                print(f"seed {args.seed} ep {epoch} step {step}/{total} L1 {loss.item():.4f} "
                      f"{(time.time() - t0) / step:.2f}s/it", flush=True)
        torch.save(model.state_dict(), OUT / f"predictor_seed{args.seed}.pt")
    print("done", flush=True)


def spearman(a, b):
    ra, rb = a.argsort(-1).argsort(-1).float(), b.argsort(-1).argsort(-1).float()
    ra, rb = ra - ra.mean(-1, keepdim=True), rb - rb.mean(-1, keepdim=True)
    return (ra * rb).sum(-1) / (ra.norm(dim=-1) * rb.norm(dim=-1))


@torch.no_grad()
def evaluate(args):
    data = TokenCache("val")
    models = []
    for seed in (0, 1):
        m = Predictor().cuda().eval()
        m.load_state_dict(torch.load(OUT / f"predictor_seed{seed}.pt", map_location="cuda"))
        models.append(m)
    sums = {"pred0": 0.0, "pred1": 0.0, "copy_last": 0.0, "mean": 0.0}
    rhos, n = [], 0
    for v in data.vids:
        maps = []
        for i in range(0, len(data.ctx[v]), args.batch):
            ctx = bf16(data.ctx[v][i:i + args.batch]).cuda()
            tgt = bf16(data.tgt[v][i:i + args.batch]).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = [m(ctx) for m in models]
            e = [error_map(q, tgt) for q in p]
            b = len(ctx)
            sums["pred0"] += e[0].mean((1, 2, 3)).sum().item()
            sums["pred1"] += e[1].mean((1, 2, 3)).sum().item()
            last = ctx.view(b, 4, 576, -1)[:, 3:4].float()
            sums["copy_last"] += (last - tgt.view(b, 2, 576, -1).float()).abs().mean().item() * b
            sums["mean"] += (tgt.float() - tgt.float().mean(1, keepdim=True)).abs().mean().item() * b
            rhos.append(spearman(e[0].flatten(1), e[1].flatten(1)).cpu())
            maps.append(torch.stack(e, 1).half().cpu())
            n += b
        torch.save(torch.cat(maps), OUT / f"errmap_{v}.pt")
    rho = torch.cat(rhos)
    res = {k: s / n for k, s in sums.items()}
    res.update(clips=n, spearman_mean=rho.mean().item(), spearman_p10=rho.quantile(0.1).item())
    res["pass"] = bool(max(res["pred0"], res["pred1"]) < min(res["copy_last"], res["mean"])
                       and res["spearman_mean"] > 0.8)
    (OUT / "eval.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    train(args) if args.mode == "train" else evaluate(args)


if __name__ == "__main__":
    main()
