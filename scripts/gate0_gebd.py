"""Gate 0: is the normalized fast error a better boundary signal than feature difference?

Reads the GEBD val cache from `extract.py gebd --hop-frames 2 --stride 2 --no-slot`
(4 fps, 0.5 s hop). Step n's first target token covers clip seconds
[0.5 n + 2, 0.5 n + 2.5) and is the error 0.5 s past a 2 s context, so the error
signal has one sample per token at its tubelet centre 0.5 n + 2.25 (the last step
adds its second token). Feature difference compares the recall cells of
consecutive steps, whose 1 s windows are offset by 0.5 s, centred at 0.5 n + 2.5.

Signals:
    norm_mean, norm_p90   spatial mean / 90th percentile of the causal per-patch z-score
    raw_mean              spatial mean of the raw error
    featdiff              || c_n - c_{n-1} || on L2-normalized recall cells (1 s windows, plan baseline)
    featdiff_tok          the same on per-token cells (0.5 s tubelets), centred between adjacent tubelets
    random                uniform noise at the error timestamps (floor of the peak picker)

Per-patch normalization statistics are cross-fitted: only the val split was
downloaded, so videos are split into two folds by id and each fold is
normalized with the other fold's statistics. Thresholds are chosen the same way.

E3a: F1 at relative distance 0.05 against the best-matching annotator (official
GEBD protocol: greedy nearest matching, rater with the highest F1 per video).
Detections are local maxima above a threshold. Reported for all boundaries and
for boundaries after 2 s, the first time any signal exists. Because dense peaks
alone score well under this protocol (see the random signal), each case is also
scored with matched counts: every video keeps its top-k peaks, k = the median
number of boundaries over its raters, which tests placement independent of rate.
E3d: Spearman correlation between the signal and the fraction of annotators
marking a boundary within 0.25 s of each sample.

Annotations are the raw pkl: range boundaries become their midpoint, and
annotators with self-consistency f1_consis < 0.3 are dropped (approximating the
official processed file).

    python scripts/gate0_gebd.py
"""
import json
import pickle
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wcem.data.surprise import ema_zscore, prior_from  # noqa: E402
from wcem.paths import ANNOTATIONS, CACHE, CHECKPOINTS  # noqa: E402

FEAT = CACHE / "features" / "gebd"
OUT = CHECKPOINTS / "gate0"
REL_DIS = 0.05
MIN_CONSIS = 0.3
SIGNALS = ("norm_mean", "norm_p90", "raw_mean", "featdiff", "featdiff_tok", "random")


def annotations():
    raw = pickle.load(open(ANNOTATIONS / "gebd" / "k400_val_raw_annotation.pkl", "rb"))
    out = {}
    for vid, v in raw.items():
        raters = [sorted((b["start_time"] + b["end_time"]) / 2 for b in r)
                  for r, c in zip(v["substages_timestamps"], v["f1_consis"]) if c >= MIN_CONSIS]
        if raters:
            out[vid] = dict(duration=v["video_duration"], raters=raters)
    return out


def bf16(path):
    return torch.from_numpy(np.load(path).astype(np.int32) << 16).view(torch.float32)


def load(vid):
    err = np.load(FEAT / f"{vid}_err.npy").astype(np.float32)
    return err, bf16(FEAT / f"{vid}_cells.npy").flatten(1), bf16(FEAT / f"{vid}_tcells.npy").flatten(2)


def diff(c):
    c = torch.nn.functional.normalize(c, dim=-1)
    return (c[1:] - c[:-1]).norm(dim=-1).numpy()


def signals(err, z, cells, tcells):
    n = len(err)
    t_err = np.concatenate([0.5 * np.arange(n) + 2.25, [0.5 * (n - 1) + 2.75]])
    tok = lambda a: np.concatenate([a[:, 0], a[-1:, 1]])  # noqa: E731
    zf = z.numpy()
    tc = torch.cat([tcells[:, 0], tcells[-1:, 1]])
    return {
        "norm_mean": (t_err, tok(zf.mean(-1))),
        "norm_p90": (t_err, tok(np.quantile(zf, 0.9, axis=-1))),
        "raw_mean": (t_err, tok(err.mean(-1))),
        "featdiff": (0.5 * np.arange(1, n) + 2.5, diff(cells)),
        "featdiff_tok": (t_err[1:] - 0.25, diff(tc)),
        "random": (t_err, np.random.default_rng(n).random(len(t_err))),
    }


def peaks(t, s, thr):
    if len(s) == 0:
        return []
    p = np.pad(s, 1, constant_values=-np.inf)
    keep = (s >= p[:-2]) & (s > p[2:]) & (s > thr)
    return list(t[keep])


def match(gt, det, threshold):
    """Official GEBD greedy matching: true positives for one rater."""
    if not gt or not det:
        return 0
    off = np.abs(np.asarray(gt)[:, None] - np.asarray(det)[None])
    tp = 0
    for i in range(len(gt)):
        if off.shape[1] == 0:
            break
        j = off[i].argmin()
        if off[i, j] <= threshold:
            tp += 1
            off = np.delete(off, j, 1)
    return tp


def top_peaks(t, s, k):
    p = np.pad(s, 1, constant_values=-np.inf)
    idx = np.nonzero((s >= p[:-2]) & (s > p[2:]))[0]
    return sorted(t[idx[np.argsort(-s[idx])[:k]]])


def f1_eval(items, name, thr=None, t_min=0.0):
    """thr=None: each video keeps its top-k peaks, k = median boundary count over raters."""
    tp = n_pos = n_det = 0
    for vid, sig, ann in items:
        if thr is None:
            det = top_peaks(*sig[name], int(np.median([len(r) for r in ann["raters"]])))
        else:
            det = peaks(*sig[name], thr)
        best = None
        for gt in ann["raters"]:
            gt = [g for g in gt if g >= t_min]
            m = match(gt, det, REL_DIS * ann["duration"])
            p, r = m / max(len(det), 1), m / max(len(gt), 1)
            f = 2 * p * r / (p + r) if p + r > 0 else 0.0
            if best is None or f > best[0]:
                best = (f, m, len(gt), len(det))
        tp, n_pos, n_det = tp + best[1], n_pos + best[2], n_det + best[3]
    p, r = tp / max(n_det, 1), tp / max(n_pos, 1)
    return dict(f1=2 * p * r / (p + r) if p + r > 0 else 0.0, precision=p, recall=r)


def best_threshold(items, name, grid):
    scores = [f1_eval(items, name, thr)["f1"] for thr in grid]
    return grid[int(np.argmax(scores))]


def agreement(items, name):
    xs, ys, per_video = [], [], []
    for vid, sig, ann in items:
        t, s = sig[name]
        frac = np.mean([[any(abs(g - x) <= 0.25 for g in gt) for x in t] for gt in ann["raters"]], 0)
        xs.append(s)
        ys.append(frac)
        if frac.std() > 0 and s.std() > 0:
            per_video.append(spearmanr(s, frac).statistic)
    return dict(spearman_pooled=spearmanr(np.concatenate(xs), np.concatenate(ys)).statistic,
                spearman_per_video=float(np.mean(per_video)))


def main():
    ann = annotations()
    vids = sorted(v for v in ann if (FEAT / f"{v}.json").exists()
                  and json.loads((FEAT / f"{v}.json").read_text())["steps"] > 1)
    print(f"{len(vids)} extracted videos with consistent annotators", flush=True)
    fold = {v: zlib.crc32(v.encode()) % 2 for v in vids}
    data = {v: load(v) for v in vids}
    priors = [prior_from([data[v][0] for v in vids if fold[v] != k]) for k in (0, 1)]
    items = []
    for v in vids:
        err = data[v][0]
        z = ema_zscore(err, *priors[fold[v]])
        items.append((v, signals(err, z, *data[v][1:]), ann[v]))
    folds = [[it for it in items if fold[it[0]] == k] for k in (0, 1)]
    results = {"videos": len(items), "rel_dis": REL_DIS}
    for name in SIGNALS:
        pooled = np.concatenate([it[1][name][1] for it in items])
        grid = list(np.quantile(pooled, np.linspace(0.3, 0.98, 35)))
        thr = [best_threshold(folds[1 - k], name, grid) for k in (0, 1)]
        res = {}
        for label, t_min in (("all", 0.0), ("after_2s", 2.0)):
            per = [f1_eval(folds[k], name, thr[k], t_min) for k in (0, 1)]
            w = [len(f) for f in folds]
            res[label] = {m: float(np.average([p[m] for p in per], weights=w)) for m in per[0]}
            res[f"{label}_matched_count"] = f1_eval(items, name, None, t_min)
        res.update(agreement(items, name))
        results[name] = res
    print(f"{len(items)} videos | F1 at threshold (all, >2 s) | F1 matched count (all, >2 s) "
          "| Spearman (pooled, per video)")
    for name in SIGNALS:
        r = results[name]
        print(f"{name:13}" + "".join(f"{x:8.3f}" for x in (
            r["all"]["f1"], r["after_2s"]["f1"], r["all_matched_count"]["f1"], r["after_2s_matched_count"]["f1"],
            r["spearman_pooled"], r["spearman_per_video"])), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gebd.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
