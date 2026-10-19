"""E0: where does the Stage 0 error go, raw and after per-patch normalization?

For each VISOR-annotated frame in a D2 video, find the step whose span holds it
and the temporal token within the step, and read that token's error map from
the M4 feature cache. VISOR polygons are rasterised onto the same centre crop
and pooled to the 24x24 patch grid.

Per region (hands, in-contact objects, water, other annotated objects, and
unannotated background):
    raw:        error share / area share (>1 means error concentrates there), AUC
    normalized: mean z-score (causal 60 s per-patch EMA), AUC
AUC is patch error separating region patches from all other patches.
Frames are split by whether an EK100 "wash" action covers them, and by whether
the Stage 0 predictor saw the video (D2 train) or not (D2 val and D3).

    python scripts/e0_regions.py [--features data/cache/features/ek100] [--size 384]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wcem.data.surprise import ema_zscore, prior_from  # noqa: E402
from wcem.data.timeline import SIZE, step_of_time  # noqa: E402
from wcem.paths import ANNOTATIONS, CACHE, CHECKPOINTS, MANIFESTS  # noqa: E402

VISOR = next((ANNOTATIONS / "visor").glob("*/GroundTruth-SparseAnnotations/annotations"))
MAPPING = next((ANNOTATIONS / "visor").glob("*/frame_mapping.json"))
OUT = CHECKPOINTS / "stage0"
REGIONS = ("hand", "contact", "water", "object")


def frame_no(name):
    return int(name.rsplit("_", 1)[1].split(".")[0])


def masks(anns, width, height, size=SIZE):
    """Rasterise to the centre-cropped size grid, return {region: [size/16, size/16] coverage}."""
    s = size / min(width, height)
    left = (round(width * s) - size) // 2
    top = (round(height * s) - size) // 2
    contact = {a.get("in_contact_object") for a in anns if "hand" in a["name"] or "glove" in a["name"]}
    out = {}
    for region in REGIONS:
        img = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(img)
        for a in anns:
            is_hand = "hand" in a["name"] or "glove" in a["name"]
            kind = ("hand" if is_hand else "water" if "water" in a["name"]
                    else "contact" if a["id"] in contact else "object")
            if kind != region:
                continue
            for poly in a["segments"]:
                pts = [(x * s - left, y * s - top) for x, y in poly]
                if len(pts) >= 3:
                    draw.polygon(pts, fill=255)
        m = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
        out[region] = F.avg_pool2d(m[None, None], 16)[0, 0]
    return out


def auc(score, label):
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = torch.cat([pos, neg]).argsort().argsort().float()
    return ((ranks[: len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg))).item()


def region_stats(row, prefix, e, cover, share):
    bg = torch.ones_like(e, dtype=torch.bool)
    for r, c in list(cover.items()) + [("bg", None)]:
        if c is None:
            label, area = bg, bg.float().mean().item()
        else:
            c = c.flatten()
            label, area = c > 0.5, c.mean().item()
            bg &= c < 0.05
        row[f"{r}_area"] = area
        if share:
            ok = area > 1e-3 and label.any()
            row[f"{prefix}{r}_ratio"] = ((e * (label.float() if c is None else c)).sum() / e.sum()).item() / area \
                if ok else float("nan")
        else:
            row[f"{prefix}{r}_z"] = e[label].mean().item() if label.any() else float("nan")
        row[f"{prefix}{r}_auc"] = auc(e, label) if label.any() and (~label).any() else float("nan")


def frame_stats(row, e, z):
    e, z = np.asarray(e, dtype=np.float32), np.asarray(z, dtype=np.float32)
    row["raw_frame_mean"], row["raw_frame_p90"] = float(e.mean()), float(np.quantile(e, 0.9))
    row["norm_frame_mean"], row["norm_frame_p90"] = float(z.mean()), float(np.quantile(z, 0.9))


def timeline(err, z, washes, split_of):
    """Frame-level error over every token of every video, washing vs other time."""
    rows = []
    for vid, e in err.items():
        n = len(e)
        t = (np.arange(n)[:, None] + 2.25 + 0.5 * np.arange(2)[None]).ravel()
        wash = np.zeros(len(t), bool)
        for a, b in washes.get(vid, []):
            wash |= (t >= a) & (t < b)
        ef = np.asarray(e, dtype=np.float32).reshape(2 * n, -1)
        zf = z[vid].reshape(2 * n, -1).numpy()
        for seg, m in (("washing", wash), ("other", ~wash)):
            if m.any():
                rows.append(dict(split=split_of(vid), segment=seg, video=vid, seconds=m.sum() / 2,
                                 raw_frame_mean=ef[m].mean(), raw_frame_p90=np.quantile(ef[m], 0.9, axis=1).mean(),
                                 norm_frame_mean=zf[m].mean(), norm_frame_p90=np.quantile(zf[m], 0.9, axis=1).mean(),
                                 norm_frame_mean_sd=zf[m].mean(1).std()))
    df = pd.DataFrame(rows)
    cols = [c for c in df.columns if c.startswith(("raw_", "norm_"))]
    out = df.groupby(["split", "segment"]).apply(
        lambda g: pd.Series({**{c: np.average(g[c], weights=g.seconds) for c in cols},
                             "hours": g.seconds.sum() / 3600, "videos": len(g)}), include_groups=False)
    out.index = [f"{a}/{b}" for a, b in out.index]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=CACHE / "features" / "ek100")
    ap.add_argument("--size", type=int, default=SIZE)
    args = ap.parse_args()
    info = pd.read_csv(ANNOTATIONS / "ek100" / "EPIC_100_video_info.csv").set_index("video_id")
    acts = pd.concat([pd.read_csv(ANNOTATIONS / "ek100" / f"EPIC_100_{s}.csv") for s in ("train", "validation")])
    acts = acts[acts.verb == "wash"]
    to_sec = lambda ts: sum(float(x) * m for x, m in zip(ts.split(":"), (3600, 60, 1)))  # noqa: E731
    washes = {v: list(zip(g.start_timestamp.map(to_sec), g.stop_timestamp.map(to_sec)))
              for v, g in acts.groupby("video_id")}
    mapping = json.loads(MAPPING.read_text())
    d2 = {s: set((MANIFESTS / f"ek100_d2_{s}.txt").read_text().split()) for s in ("train", "val")}
    held = d2["val"] | set((MANIFESTS / "ek100_d3.txt").read_text().split())
    err = {v: np.load(args.features / f"{v}_err.npy", mmap_mode="r")
           for v in d2["train"] | held if (args.features / f"{v}.json").exists()}
    prior = prior_from([err[v] for v in d2["train"] if v in err])
    z = {v: ema_zscore(np.asarray(e, dtype=np.float32), *prior) for v, e in err.items()}
    side = args.size // 16
    rows = []
    for path in sorted(VISOR.glob("*/*.json")):
        vid = path.stem
        split = "seen" if vid in d2["train"] else "heldout" if vid in held else None
        if split is None or vid not in err:
            continue
        fps = info.loc[vid, "fps"]
        width, height = map(int, info.loc[vid, "resolution"].split("x"))
        vmap = mapping.get(vid, {})
        before = len(rows)
        for entry in json.loads(path.read_text())["video_annotations"]:
            name = entry["image"]["name"]
            rgb = frame_no(vmap.get(name, name))
            t = (rgb - 1) / fps
            n = step_of_time(t)
            if n < 0 or n >= len(err[vid]):
                continue
            token = int(t - (n + 2) >= 0.5)
            cover = masks(entry["annotations"], width, height, args.size)
            washing = any(a <= t < b for a, b in washes.get(vid, []))
            row = dict(video=vid, split=split, segment="washing" if washing else "other", frame=rgb, step=n)
            region_stats(row, "raw_", torch.from_numpy(np.asarray(err[vid][n, token], dtype=np.float32)),
                         cover, share=True)
            region_stats(row, "norm_", z[vid][n, token].reshape(side * side), cover, share=False)
            frame_stats(row, err[vid][n, token], z[vid][n, token])
            rows.append(row)
        print(f"{vid} ({split}): {len(rows) - before} frames", flush=True)
    df = pd.DataFrame(rows)
    tag = f"_{args.size}" if args.size != SIZE else ""
    df.to_csv(OUT / f"e0_regions{tag}.csv", index=False)
    cols = [c for c in df.columns if c.endswith(("_ratio", "_auc", "_z")) or "_frame_" in c]
    summary = df.groupby(["split", "segment"])[cols].mean()
    summary["frames"] = df.groupby(["split", "segment"]).size()
    summary.index = [f"{a}/{b}" for a, b in summary.index]
    print("VISOR frames:\n" + summary.T.round(3).to_string(), flush=True)
    (OUT / f"e0_summary{tag}.json").write_text(summary.round(4).to_json(indent=2))
    tl = timeline(err, z, washes, lambda v: "seen" if v in d2["train"] else "heldout")
    print("All tokens of all videos:\n" + tl.T.round(3).to_string(), flush=True)
    (OUT / f"e0_timeline{tag}.json").write_text(tl.round(4).to_json(indent=2))


if __name__ == "__main__":
    main()
