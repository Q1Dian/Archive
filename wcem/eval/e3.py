"""E3b/E3c: mean write rate on trained vs untrained activities (cross-video generalizer)."""
import json
from pathlib import Path

import numpy as np
import torch


@torch.no_grad()
def write_rate_by_bucket(model, tokens, z, surprise, log_lam, trained_mask):
    """trained_mask[t] True if step t is a 'trained' (familiar) activity context."""
    device = z.device
    lam = torch.full((1,), float(log_lam), device=device)
    t = tokens[None].to(device)
    if t.dtype == torch.bfloat16:
        t = t.float()
    out = model(t, z[None].float(), surprise[None].float(), lam)
    p = out["p_open"][0].mean(-1).cpu().numpy()
    n = min(len(p), len(trained_mask))
    p, m = p[:n], np.asarray(trained_mask, dtype=bool)[:n]
    pt = p[m]
    pu = p[~m]
    return dict(
        p_write_trained=float(np.mean(pt)) if pt.size else float("nan"),
        p_write_untrained=float(np.mean(pu)) if pu.size else float("nan"),
        n_trained=int(m.sum()),
        n_untrained=int((~m).sum()),
        trained_lt_untrained=bool(np.mean(pt) < np.mean(pu)) if pt.size and pu.size else None,
    )


def load_ek100_train_verbs(manifest_dir: Path):
    """Verb ids seen in EK100 D2 train + D3 (from per-video json meta when present)."""
    from wcem.data.cache import train_val_ids
    from wcem.paths import CACHE

    train, _ = train_val_ids()
    root = CACHE / "features" / "ek100"
    verbs = set()
    for vid in train:
        meta = root / f"{vid}.json"
        if not meta.exists():
            continue
        m = json.loads(meta.read_text())
        if "verb" in m:
            verbs.add(m["verb"])
        elif "verbs" in m:
            verbs.update(m["verbs"])
    return verbs


def activity_mask_from_meta(meta, trained_verbs):
    """Per-step mask: True when video-level activity is in trained_verbs."""
    verb = meta.get("verb") or meta.get("activity")
    if verb is None:
        return None
    return bool(verb in trained_verbs)
