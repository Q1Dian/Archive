import torch

from wcem.data.surprise import diff_prior, diff_zscore, ema_zscore, prior_from, summary, token_diffs


def test_token_diffs_streams_and_causality():
    torch.manual_seed(0)
    t = torch.randn(30, 2, 16, 64)
    t[10, 1] = t[10, 0]
    d = token_diffs(t)
    assert d.shape == (30, 2, 16)
    assert torch.allclose(d[10, 1], torch.zeros(16), atol=1e-6)
    prior = diff_prior([d])
    z = diff_zscore(d, prior)
    t2 = t.clone()
    t2[20:] += 3.0
    z2 = diff_zscore(token_diffs(t2), prior)
    assert torch.allclose(z[:20], z2[:20])


def test_zscore_is_causal():
    torch.manual_seed(0)
    e = torch.rand(20, 2, 576)
    mean, var = prior_from([e])
    z = ema_zscore(e, mean, var)
    e2 = e.clone()
    e2[10:] += 5.0
    z2 = ema_zscore(e2, mean, var)
    assert torch.allclose(z[:10], z2[:10])
    assert not torch.allclose(z[10], z2[10])


def test_stationary_input_gives_unit_scale():
    torch.manual_seed(0)
    e = torch.randn(2000, 2, 64) * 3 + 7
    z = ema_zscore(e, torch.full((64,), 7.0), torch.full((64,), 9.0))
    tail = z[500:].flatten()
    assert abs(tail.mean().item()) < 0.1
    assert abs(tail.std().item() - 1.0) < 0.1


def test_summary_shape_and_values():
    z = torch.zeros(3, 2, 576)
    z[1, 0, :100] = 3.0
    s = summary(z)
    assert s.shape == (3, 4)
    assert s[0].abs().sum() == 0
    assert torch.isclose(s[1, 3], torch.tensor(100 / 1152))
