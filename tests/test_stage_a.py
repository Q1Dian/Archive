import math

import torch

from wcem.eval.stage_a import lag_bin, memory_gain, sigma_clamp_frac, written_key_cosine, writes_per_minute


def test_writes_per_minute_is_sixty_writes_per_step():
    assert writes_per_minute(0.1) == 6.0


def test_lag_bin_edges():
    assert lag_bin(8) == "8-16"
    assert lag_bin(16) == "16-32"
    assert lag_bin(64) == "64-128"
    assert lag_bin(300) == "256+"


def test_memory_gain_by_bin():
    nll_own = torch.tensor([1.0, 2.0, 3.0, 4.0])
    nll_swap = torch.tensor([2.0, 4.0, 3.0, 8.0])
    lags = torch.tensor([10, 20, 10, 80])
    g = memory_gain(nll_own, nll_swap, lags)
    assert math.isclose(g["8-16"], 0.5)
    assert math.isclose(g["16-32"], 2.0)
    assert math.isclose(g["64-128"], 4.0)
    assert "32-64" not in g


def test_written_key_cosine_and_sigma_clamp():
    k = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])  # [B, N=1, H=2, D]
    p = torch.tensor([[[1.0, 1.0]]])
    assert abs(written_key_cosine(k, p) - 0.0) < 1e-5
    ls = torch.tensor([-3.0, 0.0, 2.0, 2.0])
    assert abs(sigma_clamp_frac(ls, (-3.0, 2.0)) - 0.75) < 1e-6
