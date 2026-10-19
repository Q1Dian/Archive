"""M4: single-pass feature extraction, per video as transcodes arrive.

One full-clip encoder pass, one context-only pass and the Stage 0 predictor per
clip. Per step (clip n -> step n, temporal tokens 4-5), stores:

    slot  [N, 144, 1024] bf16   12x12 tokens averaged over the 2 temporal tokens
    cells [N, 16, 1024]  bf16   4x4 recall cells (6x6 patches x 2 temporal tokens)
    err   [N, 2, 576]    fp16   raw per-patch Stage 0 error
    full  [N, 2, 576, 1024] bf16 only with --full (slot-resolution ablation)
    tcells [N, 2, 16, 1024] bf16 only with --token-cells (recall cells per temporal token)

bf16 arrays are stored as uint16 .npy. Outputs go to data/cache/features/<dataset>/.

    python scripts/extract.py ek100 --manifest data/manifests/ek100_d2.txt --follow
    python scripts/extract.py gebd --hop-frames 2 --stride 2 --no-slot --token-cells
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wcem.data.timeline import CLIP_FRAMES, ClipReader  # noqa: E402
from wcem.data.assembly101 import manifest_transcodes  # noqa: E402
from wcem.paths import CACHE, CHECKPOINTS, TRANSCODED  # noqa: E402
from wcem.perception.encoder import Encoder  # noqa: E402
from wcem.perception.predictor import Predictor, error_map  # noqa: E402

BATCH = 16


def to_u16(x):
    return x.to(torch.bfloat16).view(torch.uint16).cpu().numpy()


def videos(dataset, manifest):
    root = TRANSCODED / dataset
    if manifest is not None:
        if dataset == "assembly101":
            return manifest_transcodes(manifest)
        ids = manifest.read_text().split()
        return {v: root / f"{v}.mp4" for v in ids}
    return {str(p.relative_to(root).with_suffix("")).replace("/", "__"): p
            for p in sorted(root.rglob("*.mp4")) if not p.name.endswith(".tmp.mp4")}


@torch.no_grad()
def extract(enc, pred, src, out, vid, hop, slot, full, stride=1, token_cells=False):
    done = out / f"{vid}.json"
    if done.exists():
        return False
    reader = ClipReader(src, stride=stride)
    total = max(0, (reader.num_frames - CLIP_FRAMES) // hop + 1)
    if total == 0:
        done.write_text(json.dumps({"video": vid, "steps": 0, "hop_frames": hop, "stride": stride}))
        return True
    arrays = {
        "cells": ((total, 16, 1024), np.uint16),
        "err": ((total, 2, 576), np.float16),
    }
    if slot:
        arrays["slot"] = ((total, 144, 1024), np.uint16)
    if full:
        arrays["full"] = ((total, 2, 576, 1024), np.uint16)
    if token_cells:
        arrays["tcells"] = ((total, 2, 16, 1024), np.uint16)
    mm = {k: np.lib.format.open_memmap(out / f"{vid}_{k}.npy", "w+", dt, shape) for k, (shape, dt) in arrays.items()}
    for i in range(0, total, BATCH):
        b = min(BATCH, total - i)
        a = i * hop
        frames = reader.frames(a, a + (b - 1) * hop + CLIP_FRAMES)
        sel = torch.arange(b)[:, None] * hop + torch.arange(CLIP_FRAMES)
        x = frames[sel].permute(0, 2, 1, 3, 4).cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = enc(x)[:, 4:6]
            ctx = enc.context(x).flatten(1, 3)
            p = pred(ctx)
        tgt = f.flatten(1, 3)
        mm["err"][i:i + b] = error_map(p, tgt).flatten(2).half().cpu().numpy()
        grid = f.float().mean(1).permute(0, 3, 1, 2)
        mm["cells"][i:i + b] = to_u16(F.avg_pool2d(grid, 6).flatten(2).transpose(1, 2))
        if slot:
            mm["slot"][i:i + b] = to_u16(F.avg_pool2d(grid, 2).flatten(2).transpose(1, 2))
        if full:
            mm["full"][i:i + b] = to_u16(f.flatten(2, 3))
        if token_cells:
            tg = f.float().flatten(0, 1).permute(0, 3, 1, 2)
            mm["tcells"][i:i + b] = to_u16(F.avg_pool2d(tg, 6).flatten(2).transpose(1, 2)).reshape(b, 2, 16, -1)
    for m in mm.values():
        m.flush()
    done.write_text(json.dumps({"video": vid, "steps": total, "hop_frames": hop, "stride": stride,
                                "arrays": list(mm)}))
    return True


@torch.no_grad()
def add_token_cells(enc, src, out, vid):
    """Backfill tcells for an extracted video: full-clip encoder pass only."""
    meta = json.loads((out / f"{vid}.json").read_text())
    total, hop, stride = meta["steps"], meta["hop_frames"], meta.get("stride", 1)
    if total == 0 or "tcells" in meta.get("arrays", []):
        return False
    reader = ClipReader(src, stride=stride)
    mm = np.lib.format.open_memmap(out / f"{vid}_tcells.npy", "w+", np.uint16, (total, 2, 16, 1024))
    for i in range(0, total, BATCH):
        b = min(BATCH, total - i)
        a = i * hop
        frames = reader.frames(a, a + (b - 1) * hop + CLIP_FRAMES)
        sel = torch.arange(b)[:, None] * hop + torch.arange(CLIP_FRAMES)
        x = frames[sel].permute(0, 2, 1, 3, 4).cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = enc(x)[:, 4:6]
        tg = f.float().flatten(0, 1).permute(0, 3, 1, 2)
        mm[i:i + b] = to_u16(F.avg_pool2d(tg, 6).flatten(2).transpose(1, 2)).reshape(b, 2, 16, -1)
    mm.flush()
    meta["arrays"] = meta.get("arrays", []) + ["tcells"]
    (out / f"{vid}.json").write_text(json.dumps(meta))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--hop-frames", type=int, default=4)
    ap.add_argument("--stride", type=int, default=1, help="source frames per 4 fps frame (2 for 8 fps GEBD)")
    ap.add_argument("--no-slot", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--token-cells", action="store_true", help="also store recall cells per temporal token")
    ap.add_argument("--add-token-cells", action="store_true", help="backfill tcells for already-extracted videos")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--follow", action="store_true", help="keep polling for new transcodes (needs --manifest)")
    ap.add_argument("--predictor", type=Path, default=CHECKPOINTS / "stage0" / "predictor_seed0.pt")
    args = ap.parse_args()
    out = args.out or CACHE / "features" / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    enc = Encoder().cuda()
    if args.add_token_cells:
        vids = videos(args.dataset, args.manifest)
        todo = [v for v in vids if (out / f"{v}.json").exists()]
        for k, v in enumerate(todo):
            t0 = time.time()
            if add_token_cells(enc, vids[v], out, v):
                print(f"token cells {v} in {time.time() - t0:.0f}s ({k + 1}/{len(todo)})", flush=True)
        return
    pred = Predictor().cuda().eval()
    pred.load_state_dict(torch.load(args.predictor, map_location="cuda"))
    while True:
        vids = videos(args.dataset, args.manifest)
        pending = {v: p for v, p in vids.items() if not (out / f"{v}.json").exists()}
        for v, p in list(pending.items()):
            if not p.exists():
                continue
            t0 = time.time()
            extract(enc, pred, p, out, v, args.hop_frames, not args.no_slot, args.full, args.stride, args.token_cells)
            del pending[v]
            print(f"extracted {v} in {time.time() - t0:.0f}s ({len(pending)} left)", flush=True)
        if not pending or not args.follow:
            break
        time.sleep(120)
    print(f"{args.dataset}: extraction complete" if not pending else f"{len(pending)} videos not yet available",
          flush=True)


if __name__ == "__main__":
    main()
