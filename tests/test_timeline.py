import torch

from wcem.data.timeline import CLIP_FRAMES, Step, num_steps, resize_center, step_of_time


def test_step_is_tokens_4_5():
    s = Step(10)
    assert s.clip_seconds == (10.0, 14.0)
    assert s.span_seconds == (12.0, 13.0)
    assert s.span_frames == (48, 52)


def test_clip_n_minus_2_ends_when_step_n_begins():
    for n in range(2, 50):
        assert Step(n - 2).clip_seconds[1] == Step(n).span_seconds[0]


def test_clip_n_minus_1_contains_step_n():
    # this is why sequence keys must come from x_{n-2}, not x_{n-1}
    for n in range(1, 50):
        a, b = Step(n - 1).clip_seconds
        sa, sb = Step(n).span_seconds
        assert a <= sa and sb <= b


def test_step_of_time_roundtrip():
    for n in range(0, 100):
        a, b = Step(n).span_seconds
        assert step_of_time(a) == n
        assert step_of_time((a + b) / 2) == n


def test_num_steps():
    assert num_steps(CLIP_FRAMES - 1) == 0
    assert num_steps(CLIP_FRAMES) == 1
    assert num_steps(CLIP_FRAMES + 4) == 2


def test_resize_center():
    f = torch.randint(0, 255, (3, 3, 384, 683), dtype=torch.uint8)
    out = resize_center(f)
    assert out.shape == (3, 3, 384, 384)
    assert torch.allclose(out, f[:, :, :, 149:533].float() / 255)
    assert resize_center(f, mode="squash").shape == (3, 3, 384, 384)
