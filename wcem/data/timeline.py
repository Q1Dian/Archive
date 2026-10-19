"""Clip and step timeline.

Transcoded videos are at FPS = 4, so frame i is at t = i / 4 s.
Clip n covers frames [4n, 4n + 16): seconds [n, n + 4).
With tubelet 2, the clip has 8 temporal tokens of 0.5 s each.
Step n is temporal tokens 4-5 of clip n: seconds [n + 2, n + 3).
So step n's content is first visible in clip n - 1 (as its tokens 6-7), and
clip n - 2 ends at n + 2, exactly when step n begins.
"""
from dataclasses import dataclass

import numpy as np
import torch
from torchcodec.decoders import VideoDecoder

FPS = 4
CLIP_FRAMES = 16
HOP_FRAMES = 4
TUBELET = 2
T_TOKENS = CLIP_FRAMES // TUBELET
STEP_TOKENS = (4, 5)
SIZE = 384


@dataclass(frozen=True)
class Step:
    n: int

    @property
    def clip_frames(self):
        return HOP_FRAMES * self.n, HOP_FRAMES * self.n + CLIP_FRAMES

    @property
    def clip_seconds(self):
        a, b = self.clip_frames
        return a / FPS, b / FPS

    @property
    def span_frames(self):
        a = HOP_FRAMES * self.n + STEP_TOKENS[0] * TUBELET
        return a, a + len(STEP_TOKENS) * TUBELET

    @property
    def span_seconds(self):
        a, b = self.span_frames
        return a / FPS, b / FPS


def num_steps(num_frames):
    return max(0, (num_frames - CLIP_FRAMES) // HOP_FRAMES + 1)


def step_of_time(t):
    """Index of the step whose span contains video time t (seconds)."""
    return int(np.floor(t - STEP_TOKENS[0] * TUBELET / FPS))


def _resize(x, h, w):
    return torch.nn.functional.interpolate(x, size=(h, w), mode="bilinear", antialias=True, align_corners=False)


def resize_center(frames, size=SIZE, mode="crop"):
    """uint8 [T, C, H, W] -> float [T, C, size, size] in [0, 1].

    mode="crop": resize short side to `size`, then center crop.
    mode="squash": resize both sides to `size`.
    """
    x = frames.float() / 255.0
    _, _, h, w = x.shape
    if mode == "squash":
        return _resize(x, size, size)
    s = size / min(h, w)
    nh, nw = max(size, round(h * s)), max(size, round(w * s))
    if (nh, nw) != (h, w):
        x = _resize(x, nh, nw)
    top, left = (nh - size) // 2, (nw - size) // 2
    return x[:, :, top:top + size, left:left + size]


class ClipReader:
    """Sequential clip reader over a transcoded 4 fps video.

    stride > 1 reads a higher-rate transcode (e.g. 8 fps GEBD with stride 2) as 4 fps:
    frame indices are in the 4 fps timeline, frame i is source frame i * stride.
    """

    def __init__(self, path, size=SIZE, mode="crop", device="cpu", stride=1):
        self.dec = VideoDecoder(str(path), device=device)
        self.stride = stride
        self.num_frames = -(-self.dec.metadata.num_frames // stride)
        self.size, self.mode = size, mode

    def __len__(self):
        return num_steps(self.num_frames)

    def frames(self, a, b):
        s = self.stride
        f = self.dec.get_frames_in_range(a * s, min(b * s, self.dec.metadata.num_frames), s).data
        return resize_center(f, self.size, self.mode)

    def clip(self, n):
        a, b = Step(n).clip_frames
        return self.frames(a, b)

    def clips(self, start, count):
        """Clips start .. start + count - 1 as [count, C, 16, H, W], decoding each frame once."""
        count = min(count, len(self) - start)
        a = HOP_FRAMES * start
        b = HOP_FRAMES * (start + count - 1) + CLIP_FRAMES
        f = self.frames(a, b)
        idx = torch.arange(count)[:, None] * HOP_FRAMES + torch.arange(CLIP_FRAMES)[None]
        return f[idx].permute(0, 2, 1, 3, 4)
