import json

import numpy as np
import torch

from wcem.data.cache import FeatureVideo, WindowDataset, bf16_u16, cached_batches, train_val_ids
from wcem.paths import MANIFESTS


def _video(root, vid, n, token_fill=1.0, z_fill=0.5, s_fill=0.0):
    np.save(root / f"{vid}_slot.npy", bf16_u16(torch.full((n, 4, 8), token_fill)))
    np.save(root / f"{vid}_z.npy", np.full((n, 2, 4), z_fill, np.float16))
    np.save(root / f"{vid}_surprise.npy", np.full((n, 4), s_fill, np.float32))
    np.save(root / f"{vid}_trig.npy", np.zeros((n, 5), np.float32))
    (root / f"{vid}.json").write_text(json.dumps({"video": vid, "steps": n, "arrays": ["slot"]}))


def test_window_dataset_samples_fixed_length(tmp_path):
    _video(tmp_path, "a", 20, token_fill=1.0)
    _video(tmp_path, "b", 8, token_fill=2.0)
    ds = WindowDataset(tmp_path, ["a", "b"], length=10)
    assert len(ds) == 11  # starts 0..10 on a; b too short
    tokens, z, surprise, vid, start = ds[0]
    assert tokens.shape == (10, 4, 8) and z.shape == (10, 2, 4) and surprise.shape == (10, 4)
    assert vid == "a" and start == 0
    assert torch.allclose(tokens.float(), torch.ones(10, 4, 8))


def test_window_dataset_is_deterministic_for_an_index(tmp_path):
    _video(tmp_path, "a", 15)
    ds = WindowDataset(tmp_path, ["a"], length=10)
    a, b = ds[3], ds[3]
    assert a[4] == 3
    assert torch.equal(a[0], b[0])


def test_cached_batches_samples_from_resident_video(tmp_path):
    _video(tmp_path, "a", 20, token_fill=1.0)
    _video(tmp_path, "b", 8, token_fill=2.0)
    tokens, z, surprise = next(cached_batches(tmp_path, ["a", "b"], length=10, batch=2, cache_gb=1, seed=0))
    assert tokens.shape == (2, 10, 4, 8)
    assert z.shape == (2, 10, 2, 4) and surprise.shape == (2, 10, 4)
    assert torch.allclose(tokens.float(), torch.ones_like(tokens.float()))


def test_train_val_ids_hold_out_d2_val():
    train, val = train_val_ids()
    d2v = set((MANIFESTS / "ek100_d2_val.txt").read_text().split())
    d2t = set((MANIFESTS / "ek100_d2_train.txt").read_text().split())
    d3 = set((MANIFESTS / "ek100_d3.txt").read_text().split())
    assert d2v <= val and not (d2v & train)
    assert d2t <= train and d3 <= train
