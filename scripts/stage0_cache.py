"""Stage 0 token cache: one encoder pass per clip at a 2 s hop over the D2 subset.

Per clip, stores context tokens (temporal 0-3, encoded alone) and the full-clip
tokens of step n (temporal 4-5), bf16, as per-video .npy memmaps. With
--follow it keeps polling for newly transcoded videos until the manifest is done.

    python scripts/stage0_cache.py data/manifests/ek100_d2.txt --follow
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wcem.data.timeline import CLIP_FRAMES, HOP_FRAMES, ClipReader  # noqa: E402
from wcem.paths import CACHE, TRANSCODED  # noqa: E402
from wcem.perception.encoder import Encoder  # noqa: E402

OUT = CACHE / "stage0"
HOP_STEPS = 2
BATCH = 16


def cache_video(enc, vid):
    src = TRANSCODED / "ek100" / f"{vid}.mp4"
    done = OUT / f"{vid}.json"
    if done.exists():
        return False
    reader = ClipReader(src)
    steps = list(range(0, len(reader), HOP_STEPS))
    n = len(steps)
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = np.lib.format.open_memmap(OUT / f"{vid}_ctx.npy", "w+", np.uint16, (n, 4 * 576, 1024))
    tgt = np.lib.format.open_memmap(OUT / f"{vid}_tgt.npy", "w+", np.uint16, (n, 2 * 576, 1024))
    for i in range(0, n, BATCH):
        idx = steps[i:i + BATCH]
        a = HOP_FRAMES * idx[0]
        frames = reader.frames(a, HOP_FRAMES * idx[-1] + CLIP_FRAMES)
        sel = (torch.tensor(idx)[:, None] * HOP_FRAMES - a) + torch.arange(CLIP_FRAMES)
        x = frames[sel].permute(0, 2, 1, 3, 4).cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            c = enc.context(x)
            f = enc(x)[:, 4:6]
        ctx[i:i + len(idx)] = c.flatten(1, 3).to(torch.bfloat16).view(torch.uint16).cpu().numpy()
        tgt[i:i + len(idx)] = f.flatten(1, 3).to(torch.bfloat16).view(torch.uint16).cpu().numpy()
    ctx.flush()
    tgt.flush()
    done.write_text(json.dumps({"video": vid, "steps": steps}))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--follow", action="store_true")
    args = ap.parse_args()
    vids = [v for v in args.manifest.read_text().split() if v]
    enc = Encoder().cuda()
    while True:
        pending = [v for v in vids if not (OUT / f"{v}.json").exists()]
        ready = [v for v in pending if (TRANSCODED / "ek100" / f"{v}.mp4").exists()]
        for v in ready:
            t0 = time.time()
            cache_video(enc, v)
            print(f"cached {v} in {time.time() - t0:.0f}s ({len(pending) - 1} left)", flush=True)
            pending.remove(v)
        if not pending or not args.follow:
            break
        time.sleep(60)
    print("stage0 cache complete" if not pending else f"{len(pending)} videos not yet available", flush=True)


if __name__ == "__main__":
    main()
