"""Raw video -> 4 fps, 384 px short side, H.264, no audio.

Output frame i is the input frame nearest to t = i / FPS seconds, so frame
indices map to video time without an offset table.
"""
import subprocess
from pathlib import Path

from wcem.paths import FFMPEG

FPS = 4
SHORT_SIDE = 384


def transcode(src: Path, dst: Path, crf: int = 20, threads: int = 6,
              fps: int = FPS, short_side: int | None = SHORT_SIDE, hwaccel: bool = False) -> Path:
    """`short_side=None` keeps the native size (rounded to even dimensions).

    `hwaccel=True` decodes on the GPU (NVDEC); output frames are identical, CPU use is ~9x lower.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.mp4")
    if short_side is None:
        scale = "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    else:
        scale = f"scale='if(gt(iw,ih),-2,{short_side})':'if(gt(iw,ih),{short_side},-2)':flags=area"
    vf = f"fps={fps}:round=near,{scale}"
    cmd = [
        str(FFMPEG), "-nostdin", "-loglevel", "error", "-y",
        "-threads", str(threads), *(["-hwaccel", "cuda"] if hwaccel else []), "-i", str(src),
        "-an", "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
        "-g", str(4 * fps), "-pix_fmt", "yuv420p",
        str(tmp),
    ]
    subprocess.run(cmd, check=True)
    tmp.rename(dst)
    return dst
