"""E3b (GoalStep) / E3c (Assembly101): write rate trained vs untrained activities.

Requires cached features under data/cache/features/<dataset>/.

    python scripts/eval_e3.py --ckpt checkpoints/stage_b/step4000.pt --dataset assembly101
    python scripts/eval_e3.py --ckpt checkpoints/stage_b/step4000.pt --dataset goalstep
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wcem.data.cache import u16_slice  # noqa: E402
from wcem.eval.e3 import activity_mask_from_meta, load_ek100_train_verbs, write_rate_by_bucket  # noqa: E402
from wcem.models.wcem import Config, WCEM  # noqa: E402
from wcem.paths import CACHE, CHECKPOINTS, MANIFESTS  # noqa: E402
from wcem.train_util import model_config  # noqa: E402

DATASETS = ("goalstep", "assembly101")


def list_videos(root, manifest_name, dataset):
    mf = MANIFESTS / manifest_name
    if not mf.exists():
        return []
    if dataset == "assembly101":
        from wcem.data.assembly101 import manifest_cache_ids
        return [v for v in manifest_cache_ids(mf) if (root / f"{v}.json").exists()]
    return [v for v in mf.read_text().split() if (root / f"{v}.json").exists()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--dataset", choices=DATASETS, required=True)
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--log-lam", type=float, default=-4.6)
    ap.add_argument("--sigma-obs", type=float)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    root = CACHE / "features" / args.dataset
    if not root.is_dir():
        print(json.dumps(dict(status="skipped", reason=f"no cache at {root}")), flush=True)
        return
    manifest = f"{args.dataset}.txt"
    vids = list_videos(root, manifest, args.dataset)
    if not vids:
        print(json.dumps(dict(status="skipped", reason="no videos with features")), flush=True)
        return
    device = "cuda"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg") or {}
    model = WCEM(model_config(cfg) if cfg else Config()).to(device).eval()
    model.load_state_dict(ckpt["model"], strict=False)
    if args.sigma_obs is not None:
        model.cfg.sigma_obs = args.sigma_obs
    trained_verbs = load_ek100_train_verbs(MANIFESTS)
    rows = []
    for vid in sorted(vids):
        meta = json.loads((root / f"{vid}.json").read_text())
        n = int(meta["steps"])
        if n < args.length:
            continue
        fam = activity_mask_from_meta(meta, trained_verbs)
        if fam is None:
            continue
        tokens = u16_slice(root / f"{vid}_slot.npy", 0, args.length)
        z = torch.from_numpy(np.array(np.load(root / f"{vid}_z.npy", mmap_mode="r")[: args.length], copy=True))
        surprise = torch.from_numpy(np.load(root / f"{vid}_surprise.npy")[: args.length].copy())
        mask = np.full(args.length, fam, dtype=bool)
        rows.append(dict(video=vid, **write_rate_by_bucket(model, tokens, z, surprise, args.log_lam, mask)))
    if not rows:
        print(json.dumps(dict(status="skipped", reason="no labeled videos")), flush=True)
        return
    pt = float(np.nanmean([r["p_write_trained"] for r in rows]))
    pu = float(np.nanmean([r["p_write_untrained"] for r in rows]))
    tag = "e3b" if args.dataset == "goalstep" else "e3c"
    out = dict(
        eval=tag,
        ckpt=str(args.ckpt),
        dataset=args.dataset,
        videos=len(rows),
        p_write_trained=pt,
        p_write_untrained=pu,
        trained_lt_untrained=pt < pu if not (np.isnan(pt) or np.isnan(pu)) else None,
        per_video=rows,
    )
    path = args.out or (CHECKPOINTS / "gate2" / f"{tag}_{args.ckpt.stem}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in out if k != "per_video"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
