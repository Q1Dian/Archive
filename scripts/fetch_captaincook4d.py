"""CaptainCook4D: GoPro 360p videos from Box -> 4 fps transcode at native size.

Uses the official downloader (data/cc4d_downloader, github.com/CaptainCook4D/
downloader): its Box link table and download function, i.e. the same files
and layout as `download_gopro_data.py --data2d --output_dir data/raw/captaincook4d`
(public shared links, no Box login), but one recording at a time so the disk
guard and staging cap apply. Unlike `--data2d`, recordings without GoPro 360p
are recorded as missing instead of falling back to the HoloLens view. Each
raw file is checked against the duration in metadata/video_information.csv,
transcoded to 4 fps keeping the native 360p short side, and deleted.
Outputs: data/transcoded/captaincook4d/<recording>.mp4. Safe to rerun.

    python scripts/fetch_captaincook4d.py [--limit N]
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from disk_guard import wait_for_space  # noqa: E402
from fetch_common import Status, TranscodePool, probe, setup_logging, wait_for_staging  # noqa: E402
from wcem.paths import ANNOTATIONS, DATA, MANIFESTS, RAW, TRANSCODED  # noqa: E402

DOWNLOADER = DATA / "cc4d_downloader"
sys.path.insert(0, str(DOWNLOADER))
from util import Constants, download_url  # noqa: E402

OUT = TRANSCODED / "captaincook4d"
STAGING = RAW / "captaincook4d"
STAGING_CAP_GB = 15
DURATION_TOL_S = 2.0


def raw_path(rec):
    return (STAGING / Constants.CAPTAIN_COOK_4D / Constants.GOPRO / Constants.RESOLUTION_360P
            / f"{rec}_360p.mp4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcoders", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: only the first N recordings")
    args = ap.parse_args()

    log = setup_logging("captaincook4d")
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    links = json.loads((DOWNLOADER / "metadata" / "download_links.json").read_text())
    with open(ANNOTATIONS / "captaincook4d" / "metadata" / "video_information.csv") as f:
        duration = {r["recording_id"]: float(r["duration(sec)"]) for r in csv.DictReader(f)}
    recs = list(links)
    missing = [r for r in recs if not links[r].get(Constants.GOPRO_RESOLUTION_360P)]
    (MANIFESTS / "captaincook4d.txt").write_text("\n".join(recs) + "\n")
    (MANIFESTS / "captaincook4d_missing.txt").write_text("".join(f"{r}\tno gopro_360p link\n" for r in missing))
    todo = [r for r in recs if r not in missing and not (OUT / f"{r}.mp4").exists()][: args.limit]
    log.info("%d recordings, %d without GoPro 360p, %d to do (%.1f h of video)", len(recs), len(missing), len(todo),
             sum(duration.get(r, 0) for r in todo) / 3600)

    status = Status(MANIFESTS / "captaincook4d_status.csv")
    pool = TranscodePool(status, log, workers=args.transcoders, threads=args.threads, tol_s=1.0, short_side=None)
    for rec in todo:
        src = raw_path(rec)
        wait_for_staging(STAGING, STAGING_CAP_GB, log)
        wait_for_space(2, log=log.info)
        ok = False
        for attempt in range(3):
            t0 = time.time()
            if not src.exists() and download_url((links[rec][Constants.GOPRO_RESOLUTION_360P], src)) is None:
                log.warning("download %s failed (attempt %d)", rec, attempt + 1)
                time.sleep(120 * (attempt + 1))
                continue
            try:
                d = probe(src)["duration"]
            except RuntimeError as e:
                d = -1.0
                log.warning("%s unreadable: %s", rec, e)
            if rec not in duration or abs(d - duration[rec]) <= DURATION_TOL_S:
                ok = True
                break
            log.warning("%s duration %.1fs != metadata %.1fs; discarding", rec, d, duration[rec])
            src.unlink()
        if not ok:
            log.error("download failed: %s", rec)
            status.write(key=rec, status="download_failed")
            continue
        size = src.stat().st_size
        log.info("downloaded %s: %.1f MB in %.0fs (%.1f MB/s)", rec, size / 1e6, time.time() - t0,
                 size / 1e6 / max(time.time() - t0, 1))
        pool.put(rec, src, OUT / f"{rec}.mp4")
    pool.close()
    log.info("done; failed: %s", pool.failed or "none")


if __name__ == "__main__":
    main()
