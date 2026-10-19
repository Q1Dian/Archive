"""Slot-input resolution ablation on the Stage 0 cache (full-resolution step tokens, 2 s hop, D2 subset).

Compares three inputs with the same attentive probe on EK100 verb and noun of the
action covering each step (train on D2 train videos, test on D2 val videos):

    full   576 x 1024   time-averaged step tokens (reference)
    pool12 144 x 1024   2x2 spatial average pool (the M4 default)
    pca256 576 x 256    per-token channel PCA fitted on train tokens

    python scripts/probe_slot_res.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wcem.models.blocks import AttnPool  # noqa: E402
from wcem.paths import ANNOTATIONS, CACHE, CHECKPOINTS, MANIFESTS  # noqa: E402

SRC = CACHE / "stage0"
WORK = CACHE / "probe_slot_res"
OUT = CHECKPOINTS / "probe_slot_res"
N_VERB, N_NOUN = 97, 300


def labels(vid, steps, ann):
    a = ann[ann.video_id == vid]
    t = np.asarray(steps, dtype=float) + 2.5
    verb = np.full(len(t), -1)
    noun = np.full(len(t), -1)
    for s, e, v, n in zip(a.start_s, a.stop_s, a.verb_class, a.noun_class):
        hit = (t >= s) & (t < e)
        verb[hit], noun[hit] = v, n
    return verb, noun


def to_sec(ts):
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def load_split(split, ann):
    vids = [v for v in (MANIFESTS / f"ek100_d2_{split}.txt").read_text().split() if (SRC / f"{v}.json").exists()]
    items = []
    for v in vids:
        steps = json.loads((SRC / f"{v}.json").read_text())["steps"]
        verb, noun = labels(v, steps, ann)
        for i in np.nonzero(verb >= 0)[0]:
            items.append((v, int(i), int(verb[i]), int(noun[i])))
    return items


def tokens(v, i, mm):
    x = torch.from_numpy(np.ascontiguousarray(mm[v][i])).view(torch.bfloat16)
    return x.view(2, 576, 1024).float().mean(0)


@torch.no_grad()
def build(items, name, mm, pca):
    """Writes [M, T, C] bf16 arrays for the three representations."""
    WORK.mkdir(parents=True, exist_ok=True)
    shapes = {"full": (576, 1024), "pool12": (144, 1024), "pca256": (576, 256)}
    arrs = {k: np.lib.format.open_memmap(WORK / f"{name}_{k}.npy", "w+", np.uint16, (len(items), *s))
            for k, s in shapes.items()}
    mean, comp = pca
    for j in range(0, len(items), 256):
        x = torch.stack([tokens(v, i, mm) for v, i, _, _ in items[j:j + 256]]).cuda()
        b = len(x)
        grid = x.view(b, 24, 24, 1024).permute(0, 3, 1, 2)
        reps = {
            "full": x,
            "pool12": F.avg_pool2d(grid, 2).flatten(2).transpose(1, 2),
            "pca256": (x - mean) @ comp,
        }
        for k, r in reps.items():
            arrs[k][j:j + b] = r.to(torch.bfloat16).view(torch.uint16).cpu().numpy()
    for a in arrs.values():
        a.flush()


@torch.no_grad()
def fit_pca(items, mm, n_clips=2000, dim=256):
    rng = np.random.default_rng(0)
    pick = rng.choice(len(items), min(n_clips, len(items)), replace=False)
    x = torch.cat([tokens(items[p][0], items[p][1], mm)[rng.choice(576, 64, replace=False)] for p in pick]).cuda()
    mean = x.mean(0)
    cov = (x - mean).T @ (x - mean) / len(x)
    evals, evecs = torch.linalg.eigh(cov)
    comp = evecs[:, -dim:].flip(-1)
    explained = (evals[-dim:].sum() / evals.sum()).item()
    return (mean, comp), explained


class Probe(nn.Module):
    def __init__(self, n_tokens, in_dim, dim=256):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, dim))
        self.pos = nn.Parameter(torch.randn(1, n_tokens, dim) * 0.02)
        self.pool = AttnPool(dim, heads=4)
        self.head = nn.Linear(dim, N_VERB + N_NOUN)

    def forward(self, x):
        o = self.head(self.pool(self.inp(x) + self.pos))
        return o[:, :N_VERB], o[:, N_VERB:]


def run_probe(rep, train_y, val_y, epochs=15, batch=128, seed=0):
    torch.manual_seed(seed)
    tr = np.load(WORK / f"train_{rep}.npy", mmap_mode="r")
    va = np.load(WORK / f"val_{rep}.npy", mmap_mode="r")
    model = Probe(tr.shape[1], tr.shape[2]).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    steps = epochs * (len(tr) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps, pct_start=0.1)
    ty = torch.as_tensor(train_y)

    def batch_x(arr, idx):
        idx = np.sort(idx)
        return torch.from_numpy(np.ascontiguousarray(arr[idx])).view(torch.bfloat16).cuda().float(), idx

    for _ in range(epochs):
        perm = np.random.permutation(len(tr))
        for j in range(0, len(perm) - batch + 1, batch):
            x, idx = batch_x(tr, perm[j:j + batch])
            y = ty[idx].cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lv, ln = model(x)
            loss = F.cross_entropy(lv.float(), y[:, 0]) + F.cross_entropy(ln.float(), y[:, 1])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    vy = torch.as_tensor(val_y).cuda()
    hits = torch.zeros(4, device="cuda")
    with torch.no_grad():
        for j in range(0, len(va), 512):
            x, _ = batch_x(va, np.arange(j, min(j + 512, len(va))))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lv, ln = model(x)
            y = vy[j:j + len(x)]
            for k, (lg, col) in enumerate(((lv, 0), (ln, 1))):
                top5 = lg.float().topk(5, -1).indices
                hits[2 * k] += (top5[:, 0] == y[:, col]).sum()
                hits[2 * k + 1] += (top5 == y[:, col, None]).any(-1).sum()
    h = (hits / len(va)).tolist()
    return dict(verb_top1=h[0], verb_top5=h[1], noun_top1=h[2], noun_top5=h[3])


def main():
    ann = pd.concat([pd.read_csv(ANNOTATIONS / "ek100" / f"EPIC_100_{s}.csv") for s in ("train", "validation")])
    ann["start_s"] = ann.start_timestamp.map(to_sec)
    ann["stop_s"] = ann.stop_timestamp.map(to_sec)
    train, val = load_split("train", ann), load_split("val", ann)
    print(f"labelled steps: train {len(train)}, val {len(val)}", flush=True)
    vids = {v for v, *_ in train + val}
    mm = {v: np.load(SRC / f"{v}_tgt.npy", mmap_mode="r") for v in vids}
    if not (WORK / "val_pca256.npy").exists():
        pca, explained = fit_pca(train, mm)
        print(f"PCA 256 explains {explained:.3f} of token variance", flush=True)
        build(train, "train", mm, pca)
        build(val, "val", mm, pca)
    ty = np.array([(v, n) for *_, v, n in train])
    vy = np.array([(v, n) for *_, v, n in val])
    majority = dict(verb_top1=float((vy[:, 0] == np.bincount(ty[:, 0]).argmax()).mean()),
                    noun_top1=float((vy[:, 1] == np.bincount(ty[:, 1]).argmax()).mean()))
    results = {"majority": majority}
    for rep in ("pool12", "pca256", "full"):
        results[rep] = run_probe(rep, ty, vy)
        print(rep, {k: round(x, 3) for k, x in results[rep].items()}, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
