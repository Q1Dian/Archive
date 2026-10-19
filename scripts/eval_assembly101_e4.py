"""E4-style repeat thrift on Assembly101 (dupsim >= 0.85)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wcem.data.assembly101 import manifest_cache_ids
from wcem.data.cache import u16_slice
from wcem.eval.e4 import SIM_THRESH, repeat_write_stats
from wcem.models.wcem import Config, WCEM
from wcem.paths import CACHE, CHECKPOINTS, MANIFESTS
from wcem.train_util import model_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--log-lam", type=float, default=-4.6)
    ap.add_argument("--sigma-obs", type=float)
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    device = "cuda"
    feat = CACHE / "features" / "assembly101"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg") or {}
    model = WCEM(model_config(cfg) if cfg else Config()).to(device).eval()
    model.load_state_dict(ckpt["model"], strict=False)
    if args.sigma_obs is not None:
        model.cfg.sigma_obs = args.sigma_obs
    sk = cfg.get("surprise_key", "surprise")
    vids = manifest_cache_ids(MANIFESTS / "assembly101.txt")
    if args.max_videos:
        vids = vids[: args.max_videos]
    rows = []
    for vid in vids:
        meta = feat / f"{vid}.json"
        if not meta.exists():
            continue
        n = int(json.loads(meta.read_text())["steps"])
        if n < args.length:
            continue
        tokens = u16_slice(feat / f"{vid}_slot.npy", 0, args.length)
        z = torch.from_numpy(np.array(np.load(feat / f"{vid}_z.npy", mmap_mode="r")[: args.length], copy=True))
        surprise = torch.from_numpy(np.load(feat / f"{vid}_{sk}.npy")[: args.length].copy())
        dupsim = np.load(feat / f"{vid}_dupsim.npy", mmap_mode="r")[: args.length]
        rows.append(dict(video=vid, **repeat_write_stats(
            model, tokens, z.to(device), surprise.to(device), dupsim, args.log_lam, sim_thresh=SIM_THRESH)))
    pf = float(np.nanmean([r["p_write_first"] for r in rows]))
    pr = float(np.nanmean([r["p_write_repeat"] for r in rows]))
    out = dict(
        ckpt=str(args.ckpt), videos=len(rows), log_lam=args.log_lam,
        p_write_first=pf, p_write_repeat=pr,
        repeat_minus_first=pr - pf,
        repeat_lt_first=pr < pf,
        frac_videos_repeat_lt_first=float(np.mean([1.0 if r["repeat_lt_first"] else 0.0 for r in rows if r["repeat_lt_first"] is not None])),
        per_video=rows,
    )
    path = args.out or (CHECKPOINTS / "gate2" / f"e4_assembly101_{args.ckpt.stem}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in out if k != "per_video"}, indent=2))


if __name__ == "__main__":
    main()
