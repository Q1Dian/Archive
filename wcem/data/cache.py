"""Cached-feature reader for Stage A and the M8 baselines.

`surprise.npy` is the gate input: the 4-scalar summary of the causal 60 s
z-scored 0.5 s token feature difference (Gate 0). `surprise_err.npy` is the
Stage 0 error summary, kept for the M8 ablation only.
"""
import json
import random

import numpy as np
import torch
import torch.utils.data

TRIG = ("featdiff", "err_raw", "err_norm", "fd_raw", "fd_norm")


def bf16_u16(x):
    return x.to(torch.bfloat16).view(torch.uint16).cpu().numpy()


def from_u16(path, shape=None):
    a = np.load(path, mmap_mode="r")
    t = torch.from_numpy(np.ascontiguousarray(a).astype(np.int32) << 16).view(torch.float32)
    return t if shape is None else t.view(*shape)


def train_val_ids():
    """Stage A train = D2 train + D3; val = D2 val (held out for E1)."""
    from wcem.paths import MANIFESTS
    read = lambda name: set((MANIFESTS / name).read_text().split())  # noqa: E731
    val = read("ek100_d2_val.txt")
    train = read("ek100_d2_train.txt") | read("ek100_d3.txt")
    return train, val


def u16_slice(path, start, length):
    """Window of a uint16-encoded bf16 array, kept as bf16 (no float32 copy)."""
    a = np.load(path, mmap_mode="r")
    return torch.from_numpy(np.array(a[start:start + length], copy=True)).view(torch.bfloat16)


def video_catalog(root, vids, length, surprise_key="surprise"):
    """Videos long enough for one window, with the bytes a resident cache would hold."""
    rows = []
    for v in vids:
        meta = root / f"{v}.json"
        slot, z, sur = root / f"{v}_slot.npy", root / f"{v}_z.npy", root / f"{v}_{surprise_key}.npy"
        if not (meta.exists() and slot.exists() and z.exists() and sur.exists()):
            continue
        n = int(json.loads(meta.read_text())["steps"])
        if n < length:
            continue
        rows.append((v, n, slot.stat().st_size + z.stat().st_size + sur.stat().st_size))
    return rows


def _load_resident(root, vid, surprise_key, with_dupsim=False):
    tokens = torch.from_numpy(np.load(root / f"{vid}_slot.npy")).view(torch.bfloat16)
    z = torch.from_numpy(np.load(root / f"{vid}_z.npy"))
    surprise = torch.from_numpy(np.load(root / f"{vid}_{surprise_key}.npy"))
    dupsim = None
    if with_dupsim:
        dupsim = torch.from_numpy(np.load(root / f"{vid}_dupsim.npy"))
    return tokens, z, surprise, dupsim


def cached_batches(root, vids, length, batch, surprise_key="surprise", cache_gb=10, seed=0, rotate_every=64,
                   with_dupsim=False):
    """Yield batches from a rotating RAM cache of whole videos.

    Stage A windows overlap by `length - 1` steps. Reading each window from
    disk repeats the same slot file hundreds of times. Holding a few videos
    and sampling many windows from each load removes that reread.
    """
    cat = video_catalog(root, vids, length, surprise_key)
    if not cat:
        raise RuntimeError(f"no videos with at least {length} steps under {root}")
    rng = random.Random(seed)
    order = list(range(len(cat)))
    rng.shuffle(order)
    cursor = 0
    cache, used, since = [], 0, 0
    budget = int(cache_gb * (1024 ** 3))

    def pull():
        nonlocal cursor, used
        for _ in range(len(cat)):
            vid, n, nbytes = cat[order[cursor % len(cat)]]
            cursor += 1
            if any(c["vid"] == vid for c in cache):
                continue
            if cache and used + nbytes > budget:
                continue
            tokens, z, surprise, dupsim = _load_resident(root, vid, surprise_key, with_dupsim)
            nbytes_dup = (root / f"{vid}_dupsim.npy").stat().st_size if with_dupsim and dupsim is not None else 0
            cache.append(dict(vid=vid, n=n, nbytes=nbytes + nbytes_dup, tokens=tokens, z=z, surprise=surprise,
                              dupsim=dupsim))
            used += nbytes
            return True
        return False

    while True:
        while not cache or (used < int(budget * 0.85) and len(cache) < len(cat)):
            if not pull():
                break
        item = cache[rng.randrange(len(cache))]
        nwin = item["n"] - length + 1
        starts = [rng.randrange(nwin) for _ in range(batch)]
        tokens = torch.stack([item["tokens"][s:s + length] for s in starts])
        z = torch.stack([item["z"][s:s + length] for s in starts])
        surprise = torch.stack([item["surprise"][s:s + length] for s in starts])
        if with_dupsim:
            dupsim = torch.stack([item["dupsim"][s:s + length] for s in starts])
            yield tokens, z, surprise, dupsim
        else:
            yield tokens, z, surprise
        since += 1
        if since >= rotate_every and len(cache) > 1:
            old = cache.pop(0)
            used -= old["nbytes"]
            since = 0


class WindowDataset(torch.utils.data.Dataset):
    """Every start position of a `length`-step window on videos that have surprise arrays."""

    def __init__(self, root, vids, length=512, surprise_key="surprise"):
        self.root, self.length, self.surprise_key = root, length, surprise_key
        self.index = []
        for v in vids:
            meta = root / f"{v}.json"
            if not meta.exists() or not (root / f"{v}_{surprise_key}.npy").exists():
                continue
            n = int(json.loads(meta.read_text())["steps"])
            for s in range(max(0, n - length + 1)):
                self.index.append((v, s))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        vid, start = self.index[i]
        b = start + self.length
        tokens = u16_slice(self.root / f"{vid}_slot.npy", start, self.length)
        z = torch.from_numpy(np.array(np.load(self.root / f"{vid}_z.npy", mmap_mode="r")[start:b], copy=True))
        surprise = torch.from_numpy(np.load(self.root / f"{vid}_{self.surprise_key}.npy")[start:b].copy())
        return tokens, z, surprise, vid, start


class FeatureVideo:
    """One extracted video. `surprise_key` is `surprise` (gate) or `surprise_err` (ablation)."""

    def __init__(self, root, vid, surprise_key="surprise"):
        self.vid = vid
        self.meta = json.loads((root / f"{vid}.json").read_text())
        self.n = int(self.meta["steps"])
        self.tokens = from_u16(root / f"{vid}_slot.npy")
        z = np.load(root / f"{vid}_z.npy", mmap_mode="r")
        self.z = torch.from_numpy(np.array(z, copy=True)).float()
        self.surprise = torch.from_numpy(np.load(root / f"{vid}_{surprise_key}.npy")).float()
        trig = np.load(root / f"{vid}_trig.npy")
        self.trig = {name: torch.from_numpy(trig[:, i].copy()).float() for i, name in enumerate(TRIG)}

    def window(self, start, length):
        b = start + length
        return dict(tokens=self.tokens[start:b].clone(), z=self.z[start:b].clone(),
                    surprise=self.surprise[start:b].clone())
