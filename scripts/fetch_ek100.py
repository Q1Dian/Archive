"""Download -> transcode -> delete raw, for the EK100 videos in a manifest.

One downloader thread feeds a small pool of transcoders. Videos come from
Hugging Face mirrors and must match the official MD5 list; the official
data.bris server is the last fallback. Raw staging is capped and the disk
guard is checked before every download. Safe to rerun: finished videos are
skipped.

    python scripts/fetch_ek100.py data/manifests/ek100_d2.txt
"""
import argparse
import csv
import hashlib
import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from disk_guard import wait_for_space  # noqa: E402
from wcem.data.transcode import transcode  # noqa: E402
from wcem.paths import DATA, LOGS, RAW, REPO, TRANSCODED  # noqa: E402

DOWNLOADER = DATA / "ek_download_scripts" / "epic_video_downloader_verify.py"
PYTHON = REPO / ".conda" / "bin" / "python"
OUT = TRANSCODED / "ek100"
STAGING = RAW / "ek100"
STAGING_CAP_GB = 150
PER_VIDEO_NEED_GB = 20
MIRRORS = ["linxxx3/EPIC-KITCHENS", "a1raman/epic_kitchens_100", "awsaf49/epic_kitchens_100",
           "introvoyz042/epic_kitchens_100"]


def load_md5():
    out = {}
    with open(DOWNLOADER.parent / "data" / "md5.csv") as f:
        for row in csv.DictReader(f):
            if row["file_remote_path"].endswith(".MP4"):
                out[Path(row["file_remote_path"]).stem] = row["md5"]
    return out


MD5 = load_md5()


def md5sum(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def staging_gb():
    return sum(p.stat().st_size for p in STAGING.rglob("*") if p.is_file()) / 1e9


def raw_path(vid):
    return STAGING / "EPIC-KITCHENS" / vid.split("_")[0] / "videos" / f"{vid}.MP4"


MIRROR_ENV = {**os.environ, "HF_XET_CHUNK_CACHE_SIZE_BYTES": "0", "HF_XET_HIGH_PERFORMANCE": "0"}
MIRROR_CMD = (
    "import sys; from huggingface_hub import hf_hub_download; "
    "print(hf_hub_download(sys.argv[1], sys.argv[2], repo_type='dataset', local_dir=sys.argv[3]))"
)


def download_mirror(vid, repo):
    # a child process per file, so the hub client's buffers are returned to the OS after every video
    rel = f"{vid.split('_')[0]}/videos/{vid}.MP4"
    local = STAGING / "hf" / repo.replace("/", "__")
    r = subprocess.run([str(PYTHON), "-c", MIRROR_CMD, repo, rel, str(local)],
                       capture_output=True, text=True, env=MIRROR_ENV)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-300:])
    path = Path(r.stdout.strip().splitlines()[-1])
    dst = raw_path(vid)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), dst)
    return dst


def download_official(vid):
    cmd = [str(PYTHON), str(DOWNLOADER), "--specific-videos", vid,
           "--output-path", str(STAGING), "--no-md5-progress", "--max-retries", "3"]
    r = subprocess.run(cmd, cwd=DOWNLOADER.parent, capture_output=True, text=True)
    if r.returncode != 0 or not raw_path(vid).exists():
        raise RuntimeError((r.stdout + r.stderr)[-300:])
    return raw_path(vid)


def download(vid, log):
    sources = [(m, lambda m=m: download_mirror(vid, m)) for m in MIRRORS]
    sources.append(("data.bris", lambda: download_official(vid)))
    for attempt in range(3):
        for name, fn in sources:
            try:
                path = fn()
            except Exception as e:
                log.warning("download %s from %s failed: %s", vid, name, str(e)[-200:])
                continue
            if md5sum(path) == MD5[vid]:
                log.info("md5 ok %s (%s)", vid, name)
                return path
            log.warning("md5 mismatch %s from %s; discarding", vid, name)
            path.unlink()
        time.sleep(120 * (attempt + 1))
    raise RuntimeError(f"download failed: {vid}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--transcoders", type=int, default=3)
    args = ap.parse_args()

    LOGS.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(threadName)s %(message)s",
        handlers=[logging.FileHandler(LOGS / f"fetch_{args.manifest.stem}.log"), logging.StreamHandler()],
    )
    log = logging.getLogger()
    vids = [v.strip() for v in args.manifest.read_text().split() if v.strip()]
    todo = [v for v in vids if not (OUT / f"{v}.mp4").exists()]
    log.info("%d videos in manifest, %d to do", len(vids), len(todo))

    q = queue.Queue(maxsize=4)
    failed = []

    def transcoder():
        while True:
            item = q.get()
            if item is None:
                return
            vid, src = item
            try:
                t0 = time.time()
                dst = transcode(src, OUT / f"{vid}.mp4", threads=4, hwaccel=True)
                log.info("transcoded %s in %.0fs: %.1f MB -> %.1f MB", vid, time.time() - t0,
                         src.stat().st_size / 1e6, dst.stat().st_size / 1e6)
                src.unlink()
            except Exception as e:  # keep the raw file for a retry
                log.error("transcode %s failed: %s", vid, e)
                failed.append(vid)

    workers = [threading.Thread(target=transcoder, name=f"tc{i}", daemon=True) for i in range(args.transcoders)]
    for w in workers:
        w.start()

    for vid in todo:
        if raw_path(vid).exists():
            if md5sum(raw_path(vid)) == MD5[vid]:
                q.put((vid, raw_path(vid)))
                continue
            raw_path(vid).unlink()
        while staging_gb() > STAGING_CAP_GB:
            time.sleep(60)
        wait_for_space(PER_VIDEO_NEED_GB, log=log.info)
        t0 = time.time()
        try:
            src = download(vid, log)
        except RuntimeError as e:
            log.error(str(e))
            failed.append(vid)
            continue
        size = src.stat().st_size
        log.info("downloaded %s: %.2f GB in %.0fs (%.1f MB/s)", vid, size / 1e9, time.time() - t0,
                 size / 1e6 / max(time.time() - t0, 1))
        q.put((vid, src))

    for _ in workers:
        q.put(None)
    for w in workers:
        w.join()
    log.info("done; failed: %s", failed or "none")


if __name__ == "__main__":
    main()
