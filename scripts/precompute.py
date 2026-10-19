"""M4 post-pass (CPU): whitening, normalized surprise, trigger signals, near-duplicates.

Fit once on the train videos of a manifest, written to checkpoints/precompute/:
    pca.pt     mean [1024], proj [1024, 256] (eigenvectors / sqrt(eigenvalue))
    prior.pt   per-patch mean and variance of the raw error, and per-cell mean and variance of the
               token feature difference per stream (normalization start points)

Per video, next to the extract.py arrays in data/cache/features/<dataset>/:
    <vid>_z.npy         [N, 16, 256] fp16   whitened recall targets
    <vid>_surprise.npy  [N, 4] fp32         gate input: summary of the causal 60 s z-scored per-cell
                                            0.5 s token feature difference (both diffs into the step's tokens)
    <vid>_surprise_err.npy [N, 4] fp32      the same summary of the z-scored Stage 0 error (baseline)
    <vid>_trig.npy      [N, 5] fp32         featdiff ||z_n - z_{n-1}||, raw mean error, normalized mean error,
                                            raw and normalized mean token feature difference
    <vid>_dup.npy       [N, K] int32        K most similar steps more than 2 steps away (E1 near-duplicates)
    <vid>_dupsim.npy    [N, K] fp16         their cosine similarity

    python scripts/precompute.py ek100 --fit data/manifests/ek100_d2_train.txt
    python scripts/precompute.py ek100 --manifest data/manifests/ek100_d3.txt
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wcem.data.surprise import diff_prior, diff_zscore, ema_zscore, prior_from, summary, token_diffs  # noqa: E402
from wcem.paths import CACHE, CHECKPOINTS  # noqa: E402

OUT = CHECKPOINTS / "precompute"
DIM = 256
K_DUP = 8
LOCAL = 2


def bf16(path):
    return torch.from_numpy(np.load(path, mmap_mode="r").astype(np.int32) << 16).view(torch.float32)


def done(feat, vid):
    meta = feat / f"{vid}.json"
    if not meta.exists():
        return False
    meta = json.loads(meta.read_text())
    ar = meta.get("arrays", [])
    return meta["steps"] > 0 and ("tcells" in ar or "cells" in ar)


def load_token_diffs(feat, vid):
    tpath = feat / f"{vid}_tcells.npy"
    if tpath.exists():
        return token_diffs(bf16(tpath))
    cells = bf16(feat / f"{vid}_cells.npy")
    return token_diffs(cells.unsqueeze(1).expand(-1, 2, -1, -1))


def fit(feat, vids, n_steps=200):
    rng = np.random.default_rng(0)
    xs = []
    for v in vids:
        c = bf16(feat / f"{v}_cells.npy")
        xs.append(c[rng.choice(len(c), min(n_steps, len(c)), replace=False)].flatten(0, 1))
    x = torch.cat(xs).double()
    mean = x.mean(0)
    evals, evecs = torch.linalg.eigh(torch.cov((x - mean).T))
    total = evals.sum()
    evals, evecs = evals.flip(0)[:DIM], evecs.flip(1)[:, :DIM]
    print(f"PCA {DIM} on {len(x)} cells explains {evals.sum() / total:.3f} of variance", flush=True)
    pca = dict(mean=mean.float(), proj=(evecs / evals.sqrt()).float())
    prior = dict(zip(("mean", "var"), prior_from([np.load(feat / f"{v}_err.npy", mmap_mode="r") for v in vids])))
    prior["diff"] = diff_prior([token_diffs(bf16(feat / f"{v}_tcells.npy")) for v in vids])
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(pca, OUT / "pca.pt")
    torch.save(prior, OUT / "prior.pt")


def near_duplicates(z):
    f = torch.nn.functional.normalize(z.flatten(1), dim=1)
    n = len(f)
    k = min(K_DUP, max(n - 2 * LOCAL - 1, 0))
    idx = np.zeros((n, K_DUP), np.int32) - 1
    sim = np.zeros((n, K_DUP), np.float16)
    if k == 0:
        return idx, sim
    for a in range(0, n, 1024):
        s = f[a:a + 1024] @ f.T
        rows = torch.arange(a, min(a + 1024, n))[:, None]
        s[(torch.arange(n)[None] - rows).abs() <= LOCAL] = -2
        top = s.topk(k, dim=1)
        idx[a:a + len(s), :k] = top.indices.numpy()
        sim[a:a + len(s), :k] = top.values.numpy()
    return idx, sim


@torch.no_grad()
def process(feat, vid, pca, prior):
    cells = bf16(feat / f"{vid}_cells.npy")
    err = np.load(feat / f"{vid}_err.npy").astype(np.float32)
    z = (cells - pca["mean"]) @ pca["proj"]
    zs = ema_zscore(err, prior["mean"], prior["var"])
    d = load_token_diffs(feat, vid)
    dz = diff_zscore(d, prior["diff"])
    fd = torch.cat([torch.zeros(1), (z[1:] - z[:-1]).flatten(1).norm(dim=1)])
    trig = torch.stack([fd, torch.from_numpy(err).flatten(1).mean(1), zs.flatten(1).mean(1),
                        d.flatten(1).mean(1), dz.flatten(1).mean(1)], 1)
    dup, dupsim = near_duplicates(z)
    np.save(feat / f"{vid}_z.npy", z.half().numpy())
    np.save(feat / f"{vid}_surprise.npy", summary(dz).numpy())
    np.save(feat / f"{vid}_surprise_err.npy", summary(zs).numpy())
    np.save(feat / f"{vid}_trig.npy", trig.numpy())
    np.save(feat / f"{vid}_dup.npy", dup)
    np.save(feat / f"{vid}_dupsim.npy", dupsim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--fit", type=Path, help="manifest of train videos to fit PCA and the error prior on")
    ap.add_argument("--manifest", type=Path, help="videos to process (default: every extracted video)")
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()
    feat = CACHE / "features" / args.dataset
    torch.set_num_threads(4)
    if args.fit:
        fit(feat, [v for v in args.fit.read_text().split() if done(feat, v)])
    pca, prior = torch.load(OUT / "pca.pt"), torch.load(OUT / "prior.pt")
    if args.manifest:
        if args.dataset == "assembly101":
            from wcem.data.assembly101 import manifest_cache_ids  # noqa: E402
            vids = manifest_cache_ids(args.manifest)
        else:
            vids = args.manifest.read_text().split()
    else:
        vids = sorted(p.stem for p in feat.glob("*.json"))
    vids = [v for v in vids if done(feat, v) and (args.redo or not (feat / f"{v}_dupsim.npy").exists())]
    for i, v in enumerate(vids):
        process(feat, v, pca, prior)
        print(f"{v} ({i + 1}/{len(vids)})", flush=True)


if __name__ == "__main__":
    main()
