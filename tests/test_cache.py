import numpy as np
import torch

from wcem.data.cache import FeatureVideo, TRIG, bf16_u16
from wcem.eval.triggers import writes_at_rate


def _write_video(root, vid, n=8, surprise=None, surprise_err=None, trig=None):
    slot = torch.randn(n, 144, 8)
    z = torch.randn(n, 16, 4).half()
    np.save(root / f"{vid}_slot.npy", bf16_u16(slot))
    np.save(root / f"{vid}_z.npy", z.numpy())
    np.save(root / f"{vid}_surprise.npy", np.full((n, 4), 1.0, np.float32) if surprise is None else surprise)
    np.save(root / f"{vid}_surprise_err.npy",
            np.full((n, 4), 9.0, np.float32) if surprise_err is None else surprise_err)
    t = np.stack([np.arange(n, dtype=np.float32) + k for k in range(5)], 1) if trig is None else trig
    np.save(root / f"{vid}_trig.npy", t)
    (root / f"{vid}.json").write_text('{"video":"%s","steps":%d,"arrays":["slot","z"]}' % (vid, n))


def test_feature_video_loads_fd_surprise_not_error(tmp_path):
    _write_video(tmp_path, "v")
    vid = FeatureVideo(tmp_path, "v")
    assert vid.surprise[0, 0] == 1.0
    assert vid.n == 8
    win = vid.window(2, 3)
    assert win["tokens"].shape == (3, 144, 8)
    assert win["surprise"].shape == (3, 4)
    assert torch.allclose(win["surprise"], torch.ones(3, 4))


def test_feature_video_can_load_error_surprise_for_ablation(tmp_path):
    _write_video(tmp_path, "v")
    vid = FeatureVideo(tmp_path, "v", surprise_key="surprise_err")
    assert vid.surprise[0, 0] == 9.0


def test_trig_columns_match_precompute_order(tmp_path):
    _write_video(tmp_path, "v")
    vid = FeatureVideo(tmp_path, "v")
    assert TRIG == ("featdiff", "err_raw", "err_norm", "fd_raw", "fd_norm")
    assert vid.trig["fd_norm"][3] == 7.0  # arange + 4


def test_writes_at_rate_keeps_top_peaks():
    s = torch.tensor([0.0, 3.0, 0.1, 2.0, 0.2, 4.0, 0.0])
    w = writes_at_rate(s, rate=2 / 7)
    assert w.sum() == 2
    assert w.tolist() == [False, True, False, False, False, True, False]
    assert writes_at_rate(s, rate=0).sum() == 0
