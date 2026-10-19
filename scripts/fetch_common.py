"""Pieces shared by the fetch_<dataset>.py download -> transcode -> delete jobs.

Same pattern as fetch_ek100.py: the main thread downloads, a small pool of
transcoder threads converts, every output is checked with ffprobe and the raw
file is deleted once the output looks right. Results go to a per-dataset
status CSV so reruns and reports can see what failed.
"""
import csv
import json
import logging
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wcem.data.transcode import transcode  # noqa: E402
from wcem.paths import FFMPEG, LOGS  # noqa: E402

FFPROBE = FFMPEG.parent / "ffprobe"
STATUS_FIELDS = ["key", "status", "src_mb", "out_mb", "src_s", "out_s", "out_frames", "fps", "width", "height",
                 "transcode_s", "note"]


def setup_logging(name):
    LOGS.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(threadName)s %(message)s",
        handlers=[logging.FileHandler(LOGS / f"fetch_{name}.log"), logging.StreamHandler()],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return logging.getLogger()


def dir_gb(path):
    path = Path(path)
    if not path.exists():
        return 0.0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e9


def wait_for_staging(path, cap_gb, log, poll_s=60):
    while dir_gb(path) > cap_gb:
        log.info("staging %s above %d GB; waiting", path, cap_gb)
        time.sleep(poll_s)


def probe(path):
    """fps, size, duration and packet count of the first video stream.

    The duration is the video stream's own, not the container's, which can be
    longer when the audio track outlasts the video.
    """
    cmd = [str(FFPROBE), "-v", "error", "-select_streams", "v:0", "-count_packets",
           "-show_entries", "stream=width,height,avg_frame_rate,nb_read_packets,duration:format=duration",
           "-of", "json", str(path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {r.stderr.strip()[-200:]}")
    d = json.loads(r.stdout)
    if not d.get("streams"):
        raise RuntimeError(f"no video stream in {path}")
    s = d["streams"][0]
    num, den = s.get("avg_frame_rate", "0/1").split("/")
    return {
        "width": int(s["width"]), "height": int(s["height"]),
        "fps": float(num) / float(den) if float(den) else 0.0,
        "frames": int(s.get("nb_read_packets", 0)),
        "duration": float(s.get("duration") or d.get("format", {}).get("duration", 0.0)),
    }


class Status:
    """Append-only CSV of per-item results; the last row for a key wins."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        if not self.path.exists():
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(STATUS_FIELDS)

    def write(self, **row):
        with self.lock, open(self.path, "a", newline="") as f:
            csv.DictWriter(f, STATUS_FIELDS).writerow({k: row.get(k, "") for k in STATUS_FIELDS})

    def read(self):
        with open(self.path) as f:
            return {r["key"]: r for r in csv.DictReader(f)}


def http_download(url, dst, log, retries=5, timeout=(30, 300), chunk=1 << 22):
    """Resumable GET into dst (via dst.part); checks the byte count against Content-Length."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    for attempt in range(retries):
        have = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, stream=True, timeout=timeout, headers=headers) as r:
                if have and r.status_code != 206:
                    have = 0
                    part.unlink(missing_ok=True)
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0)) + have
                with open(part, "ab" if have else "wb") as f:
                    for b in r.iter_content(chunk):
                        f.write(b)
            if total and part.stat().st_size != total:
                raise IOError(f"short read {part.stat().st_size}/{total}")
            part.rename(dst)
            return dst
        except Exception as e:
            log.warning("download %s failed (attempt %d): %s", url, attempt + 1, str(e)[-200:])
            time.sleep(30 * (attempt + 1))
    raise RuntimeError(f"download failed: {url}")


class TranscodePool:
    """Threads that transcode (key, src, dst) items, verify them with ffprobe and delete src.

    An output is accepted when it has the requested fps and its duration is
    within `tol_s` of the source; otherwise it is marked "suspect" but kept.
    A source that ffmpeg cannot read is marked "corrupt" and deleted when
    `delete_bad` is set, else kept for a retry.
    """

    def __init__(self, status, log, workers=2, threads=6, delete_bad=False, tol_s=1.0, **tc_kwargs):
        self.status, self.log, self.delete_bad, self.tol_s = status, log, delete_bad, tol_s
        self.threads, self.tc_kwargs = threads, tc_kwargs
        self.q = queue.Queue()
        self.failed = []
        self.video_s = 0.0
        self.out_bytes = 0
        self.lock = threading.Lock()
        self.workers = [threading.Thread(target=self._run, name=f"tc{i}", daemon=True) for i in range(workers)]
        for w in self.workers:
            w.start()

    def put(self, key, src, dst):
        self.q.put((key, Path(src), Path(dst)))

    def pending(self):
        return self.q.qsize()

    def close(self):
        for _ in self.workers:
            self.q.put(None)
        for w in self.workers:
            w.join()

    def _run(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            self._one(*item)

    def _one(self, key, src, dst):
        src_mb = src.stat().st_size / 1e6
        t0 = time.time()
        try:
            src_info = probe(src)
            transcode(src, dst, threads=self.threads, **self.tc_kwargs)
            out = probe(dst)
        except Exception as e:
            self.log.error("transcode %s failed: %s", key, str(e)[-300:])
            self.status.write(key=key, status="corrupt", src_mb=f"{src_mb:.1f}", note=str(e)[-120:].replace("\n", " "))
            with self.lock:
                self.failed.append(key)
            dst.with_suffix(".tmp.mp4").unlink(missing_ok=True)
            if self.delete_bad:
                src.unlink(missing_ok=True)
            return
        want_fps = self.tc_kwargs.get("fps", 4)
        ok = abs(out["fps"] - want_fps) < 0.01 and abs(out["duration"] - src_info["duration"]) <= self.tol_s
        status = "ok" if ok else "suspect"
        dt = time.time() - t0
        self.status.write(key=key, status=status, src_mb=f"{src_mb:.1f}", out_mb=f"{dst.stat().st_size / 1e6:.2f}",
                          src_s=f"{src_info['duration']:.2f}", out_s=f"{out['duration']:.2f}",
                          out_frames=out["frames"], fps=f"{out['fps']:.3f}", width=out["width"],
                          height=out["height"], transcode_s=f"{dt:.1f}",
                          note=f"src {src_info['width']}x{src_info['height']}@{src_info['fps']:.2f}")
        with self.lock:
            self.video_s += out["duration"]
            self.out_bytes += dst.stat().st_size
        log = self.log.info if ok else self.log.warning
        log("transcoded %s [%s] in %.0fs: %.1f MB %.0fs -> %.2f MB %.1fs %dx%d %d frames", key, status, dt, src_mb,
            src_info["duration"], dst.stat().st_size / 1e6, out["duration"], out["width"], out["height"],
            out["frames"])
        src.unlink(missing_ok=True)
