import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture(scope="module")
def enc():
    from wcem.perception.encoder import Encoder

    torch.manual_seed(0)
    return Encoder(pretrained=False).cuda()


@cuda
def test_shapes(enc):
    x = torch.rand(1, 3, 16, 384, 384, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = enc(x)
    assert out.shape == (1, 8, 24, 24, 1024)


@cuda
def test_context_equals_masked_call(enc):
    x = torch.rand(1, 3, 16, 384, 384, device="cuda")
    keep = torch.arange(4 * 24 * 24, device="cuda")[None]
    ctx = enc.context(x).flatten(1, 3)
    ref = enc.masked(x, keep)
    assert torch.allclose(ctx, ref, atol=1e-3, rtol=1e-3)


@cuda
def test_context_ignores_future_frames(enc):
    x = torch.rand(1, 3, 16, 384, 384, device="cuda")
    y = x.clone()
    y[:, :, 8:] = torch.rand_like(y[:, :, 8:])
    assert torch.equal(enc.context(x), enc.context(y))


@cuda
def test_token_order_is_time_height_width(enc):
    # changing only frames 14-15 (temporal token 7) must change token 7 the most
    x = torch.rand(1, 3, 16, 384, 384, device="cuda")
    y = x.clone()
    y[:, :, 14:] = torch.rand_like(y[:, :, 14:])
    d = (enc(x) - enc(y)).abs().mean(dim=(0, 2, 3, 4))
    assert d.argmax().item() == 7
