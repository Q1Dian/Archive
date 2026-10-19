"""Gate 0 in domain: Stage 0 error vs token feature difference on EK100 action boundaries.

Uses the EK100 feature cache (1 s hop) with per-token cells (`--token-cells` or
`--add-token-cells`). Token j = 2 n + k (step n, target token k) covers clip
seconds [n + 2 + 0.5 k, n + 2.5 + 0.5 k), centred at t_j = 2.25 + 0.5 j, so
the two tokens of consecutive steps tile time at 0.5 s.

Signals:
    err_raw       spatial mean raw Stage 0 error of token j
    err_norm      the same after causal 60 s per-patch z-scoring
    err_norm_p90  90th percentile over patches of the z-scored error
    fd_raw        || c_j - c_{j-1} || on L2-normalized per-token cells, at t_j - 0.25
    fd_norm       per-cell differences z-scored the same way (causal 60 s EMA), mean over the 16 cells
    random        uniform noise at the token times

Token 1 of a step is predicted 0.5 s further past the context than token 0,
and c_j vs c_{j-1} compares tokens from the same clip (j odd) or adjacent clips
(j even). So normalized signals keep separate statistics per token index /
parity (each stream at 1 s), and the raw signals are reported as-is together
with their per-parity means.

Ground truth: EK100 action starts and stops (train + validation csv), merged
within 0.5 s. "long" keeps only the boundaries of actions lasting at least
LONG_S seconds (entering and leaving long actions such as washing).

Metrics per video, pooled over videos:
    matched-count F1: the top-k local maxima, k = number of boundaries, greedy
                      nearest matching within TOL seconds (GEBD protocol)
    AUC:              signal separating samples within 0.5 s of a boundary from the rest

Videos: held out from the Stage 0 predictor (D2 val + D3) by default. Per-patch
and per-cell statistics are cross-fitted over two folds of the evaluated videos.

    python scripts/gate0_ek100.py [--videos heldout|seen]
"""
import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gate0_gebd import match, top_peaks  # noqa: E402
from wcem.data.surprise import ema_zscore, prior_from  # noqa: E402
from wcem.paths import ANNOTATIONS, CACHE, CHECKPOINTS, MANIFESTS  # noqa: E402

FEAT = CACHE / "features" / "ek100"
OUT = CHECKPOINTS / "gate0"
LONG_S = 4.0
TOLS = (1.0, 2.0)
SIGNALS = ("err_raw", "err_norm", "err_norm_p90", "fd_raw", "fd_norm", "random")


def to_sec(ts):
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def boundaries():
    acts = pd.concat([pd.read_csv(ANNOTATIONS / "ek100" / f"EPIC_100_{s}.csv") for s in ("train", "validation")])
    acts["a"], acts["b"] = acts.start_timestamp.map(to_sec), acts.stop_timestamp.map(to_sec)
    out = {}
    for vid, g in acts.groupby("video_id"):
        sets = {}
        for name, sel in (("all", g), ("long", g[g.b - g.a >= LONG_S])):
            t = np.sort(np.concatenate([sel.a.values, sel.b.values]))
            keep = [x for i, x in enumerate(t) if i == 0 or x - t[i - 1] > 0.5]
            sets[name] = np.asarray(keep)
        out[vid] = sets
    return out


def bf16(path):
    return torch.from_numpy(np.load(path).astype(np.int32) << 16).view(torch.float32)


def load(vid):
    err = np.load(FEAT / f"{vid}_err.npy").astype(np.float32)
    c = torch.nn.functional.normalize(bf16(FEAT / f"{vid}_tcells.npy"), dim=-1).flatten(0, 1)  # [2N, 16, D]
    full = torch.nn.functional.normalize(c.flatten(1), dim=1)
    fd = (full[1:] - full[:-1]).norm(dim=1).numpy()
    fd_cell = (c[1:] - c[:-1]).norm(dim=-1).numpy()  # [2N - 1, 16]
    return err, fd, fd_cell


def by_parity(x):
    """[2N - 1, ...] differences -> (odd j: within clip, even j: across clips), j = index + 1."""
    return x[0::2], x[1::2]


def build(vids, data, fold):
    """Signals per video, with normalization statistics from the other fold."""
    priors = {}
    for k in (0, 1):
        other = [v for v in vids if fold[v] != k]
        priors[k] = dict(
            err=[prior_from([data[v][0][:, i:i + 1] for v in other]) for i in (0, 1)],
            fd=[prior_from([by_parity(data[v][2])[p][:, None] for v in other]) for p in (0, 1)])
    out = {}
    for v in vids:
        err, fd, fd_cell = data[v]
        pr = priors[fold[v]]
        n = len(err)
        t = 2.25 + 0.5 * np.arange(2 * n)
        z = torch.stack([ema_zscore(err[:, i:i + 1], *pr["err"][i], dt=1.0)[:, 0] for i in (0, 1)], 1)
        z = z.flatten(0, 1).numpy()
        zfd = np.zeros_like(fd_cell)
        for p, part in enumerate(by_parity(fd_cell)):
            zfd[p::2] = ema_zscore(part[:, None], *pr["fd"][p], dt=1.0)[:, 0].numpy()
        out[v] = {
            "err_raw": (t, err.reshape(2 * n, -1).mean(1)),
            "err_norm": (t, z.mean(1)),
            "err_norm_p90": (t, np.quantile(z, 0.9, axis=1)),
            "fd_raw": (t[1:] - 0.25, fd),
            "fd_norm": (t[1:] - 0.25, zfd.mean(1)),
            "random": (t, np.random.default_rng(n).random(2 * n)),
        }
    return out


def auc(score, label):
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1.0
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def evaluate(sigs, gts, name, kind, tol):
    tp = n_pos = n_det = 0
    aucs = []
    for v, s in sigs.items():
        t, x = s[name]
        gt = gts[v][kind]
        gt = gt[(gt >= t[0]) & (gt <= t[-1])]
        if len(gt) == 0:
            continue
        det = top_peaks(t, x, len(gt))
        tp += match(list(gt), det, tol)
        n_pos, n_det = n_pos + len(gt), n_det + len(det)
        near = np.abs(t[:, None] - gt[None]).min(1) <= 0.5
        aucs.append(auc(x, near))
    p, r = tp / max(n_det, 1), tp / max(n_pos, 1)
    return dict(f1=2 * p * r / (p + r) if p + r else 0.0, auc=float(np.nanmean(aucs)), boundaries=n_pos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", choices=("heldout", "seen"), default="heldout")
    args = ap.parse_args()
    man = lambda name: set((MANIFESTS / name).read_text().split())  # noqa: E731
    pool = man("ek100_d2_train.txt") if args.videos == "seen" else man("ek100_d2_val.txt") | man("ek100_d3.txt")
    gts = boundaries()
    vids = sorted(v for v in pool if v in gts and (FEAT / f"{v}.json").exists()
                  and "tcells" in json.loads((FEAT / f"{v}.json").read_text()).get("arrays", []))
    fold = {v: zlib.crc32(v.encode()) % 2 for v in vids}
    data = {v: load(v) for v in vids}
    hours = sum(len(d[0]) for d in data.values()) / 3600
    print(f"{len(vids)} {args.videos} videos, {hours:.1f} h", flush=True)
    sigs = build(vids, data, fold)
    parity = dict(
        err_token0=float(np.mean([d[0][:, 0].mean() for d in data.values()])),
        err_token1=float(np.mean([d[0][:, 1].mean() for d in data.values()])),
        fd_within_clip=float(np.mean([by_parity(d[1])[0].mean() for d in data.values()])),
        fd_across_clips=float(np.mean([by_parity(d[1])[1].mean() for d in data.values()])))
    print("per-parity means:", {k: round(x, 4) for k, x in parity.items()}, flush=True)
    results = dict(videos=len(vids), hours=hours, parity=parity)
    cols = [(kind, tol) for kind in ("all", "long") for tol in TOLS]
    print(f"{'signal':13}" + "".join(f"{f'F1 {k} {t:g}s':>13}" for k, t in cols)
          + "".join(f"{f'AUC {k}':>10}" for k in ("all", "long")), flush=True)
    for name in SIGNALS:
        res = {f"{kind}_{tol:g}s": evaluate(sigs, gts, name, kind, tol) for kind, tol in cols}
        results[name] = res
        print(f"{name:13}" + "".join(f"{res[f'{k}_{t:g}s']['f1']:13.3f}" for k, t in cols)
              + "".join(f"{res[f'{k}_1s']['auc']:10.3f}" for k in ("all", "long")), flush=True)
    print("boundaries:", {k: results["random"][f"{k}_1s"]["boundaries"] for k in ("all", "long")}, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"ek100_{args.videos}.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
