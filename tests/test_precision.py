import torch

from wcem.memory.precision import fuse_explained


def test_fusion_downweights_high_variance_L():
    v = torch.ones(1, 2, 4)
    mu_L = torch.zeros_like(v)
    mu_S = torch.ones_like(v) * 0.5
    var_L = torch.tensor([[1e6, 1e6]])
    vp = fuse_explained(v, mu_L, var_L, mu_S, sigma_S=1.0)
    assert (vp - 0.5).abs().max() < 0.01
