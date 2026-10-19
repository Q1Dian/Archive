"""E2: λ sweep — trained vs uniform at matched write count (Gate 1).

    python scripts/eval_e2.py --ckpt checkpoints/stage_a_gate_tune/step8000.pt
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wcem.data.cache import train_val_ids, u16_slice  # noqa: E402
from wcem.eval.e2 import sweep_lam  # noqa: E402
from wcem.models.wcem import Config, WCEM  # noqa: E402
from wcem.paths import CACHE, CHECKPOINTS  # noqa: E402
from wcem.train_util import model_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--max-videos", type=int, default=0, help="0 = all val windows")
    ap.add_argument("--window-stride", type=int, default=0)
    ap.add_argument("--log-lams", type=float, nargs="+",
                    default=[-4.5, -4.0, -3.5, -3.0, -2.5, -2.0])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--set-sigma-q", type=float)
    ap.add_argument("--sigma-obs", type=float, help="Kalman observation noise override")
    args = ap.parse_args()
    device = "cuda"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg") or {}
    model = WCEM(model_config(cfg) if cfg else Config()).to(device).eval()
    model.load_state_dict(ckpt["model"], strict=False)
    if args.set_sigma_q is not None:
        model.memory.sigma_q = args.set_sigma_q
    if args.sigma_obs is not None:
        model.cfg.sigma_obs = args.sigma_obs
    _, val = train_val_ids()
    root = CACHE / "features" / "ek100"
    batch = []
    seen = set()
    for vid in sorted(val):
        meta = root / f"{vid}.json"
        if not meta.exists() or not (root / f"{vid}_surprise.npy").exists():
            continue
        n = int(json.loads(meta.read_text())["steps"])
        if n < args.length:
            continue
        starts = list(range(0, n - args.length + 1, args.window_stride)) if args.window_stride > 0 else [0]
        for start in starts:
            b = start + args.length
            tokens = u16_slice(root / f"{vid}_slot.npy", start, args.length)
            z = torch.from_numpy(np.array(np.load(root / f"{vid}_z.npy", mmap_mode="r")[start:b], copy=True))
            surprise = torch.from_numpy(np.load(root / f"{vid}_surprise.npy")[start:b].copy())
            batch.append((tokens, z, surprise, vid))
            seen.add(vid)
            if args.max_videos and len(batch) >= args.max_videos:
                break
        if args.max_videos and len(batch) >= args.max_videos:
            break
    tokens = torch.stack([b[0] for b in batch]).to(device).float()
    z = torch.stack([b[1] for b in batch]).to(device).float()
    surprise = torch.stack([b[2] for b in batch]).to(device).float()
    rows = sweep_lam(model, tokens, z, surprise, args.log_lams)
    wins = sum(1 for r in rows if r["beats_uniform"])
    out = dict(ckpt=str(args.ckpt), videos=len(seen), windows=len(batch),
               window_stride=args.window_stride, sigma_q=model.memory.sigma_q,
               sweep=rows, beats_uniform_count=wins, beats_uniform_all=wins == len(rows))
    stem = args.ckpt.stem
    path = args.out or (CHECKPOINTS / "gate1" / f"e2_{stem}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("beats_uniform_count", "beats_uniform_all", "sigma_q")}, indent=2))
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
