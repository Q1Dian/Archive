"""M2 throughput benchmark: full-clip and context-only passes on a real transcoded video.

    python scripts/bench_encoder.py [video.mp4]
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wcem.data.timeline import ClipReader  # noqa: E402
from wcem.paths import TRANSCODED  # noqa: E402
from wcem.perception.encoder import Encoder  # noqa: E402


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted((TRANSCODED / "ek100").glob("*.mp4"))[0]
    reader = ClipReader(path)
    print(f"{path.name}: {reader.num_frames} frames, {len(reader)} steps")

    enc = Encoder().cuda()
    t0 = time.time()
    clips = reader.clips(0, 64)
    print(f"decode+resize 64 clips: {time.time() - t0:.2f}s")

    for bs in (4, 8, 16):
        x = clips[:bs].cuda()
        for name, fn in (("full", enc), ("context", enc.context)):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                fn(x)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                reps = 5
                for _ in range(reps):
                    out = fn(x)
                torch.cuda.synchronize()
            dt = (time.time() - t0) / reps
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"bs {bs:2d} {name:8s} {bs / dt:6.1f} clips/s  peak {peak:.1f} GB  "
                  f"out {tuple(out.shape)}  finite {torch.isfinite(out).all().item()}")


if __name__ == "__main__":
    main()
