import torch

from wcem.eval.e1 import fifo_byte_cap_g, fifo_index, fifo_keep_writes, nll_by_lag, uniform_writes
from wcem.eval.triggers import writes_at_rate


def test_uniform_writes_matches_rate():
    w = uniform_writes(100, 0.1)
    assert w.sum() == 10
    assert w[9] and w[19] and not w[8]


def test_fifo_index_returns_most_recent_kept_write():
    writes = torch.zeros(20, dtype=torch.bool)
    writes[[2, 8, 15]] = True
    # query tau=12 from n=19, keep last 2 writes: 8 and 15. 15 is after 12, so 8.
    assert fifo_index(12, writes, keep=2) == 8
    assert fifo_index(16, writes, keep=2) == 15
    assert fifo_index(1, writes, keep=2) == -1


def test_nll_by_lag_groups():
    nll = torch.tensor([1.0, 3.0, 2.0, 4.0])
    lags = torch.tensor([8, 8, 64, 64])
    d = nll_by_lag(nll, lags)
    assert abs(d["8-16"] - 2.0) < 1e-6
    assert abs(d["64-128"] - 3.0) < 1e-6


def test_fifo_byte_cap_keeps_one_write_for_256kb():
    g = fifo_byte_cap_g(100, 0.2)
    assert g.sum() == 2.0  # one step, both pair gates
    g2 = fifo_keep_writes(100, 0.2, keep=3)
    assert g2.sum() == 6.0


def test_fd_trigger_exists():
    s = torch.zeros(20)
    s[5] = 3.0
    s[12] = 4.0
    w = writes_at_rate(s, 0.1)
    assert w.sum() == 2
