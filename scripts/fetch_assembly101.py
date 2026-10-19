"""Assembly101: one fixed RGB view + one egocentric view per recording -> 4 fps transcode.

Views: v4 = C10119_rgb (overhead; the best single view for fine-grained
recognition in the paper, Table 6) and e4 = HMC_84358933 / HMC_21179183
(lower headset camera; e3/e4 see the whole table, e1/e2 do not). The two
egocentric serials are the same headset position on two headset units.

Source: the official cvml-nus/assembly101 (1080p H.264, 60 fps) is gated on
Hugging Face, so it is used only when the local HF token has access.
Otherwise the ungated pablovela5620/assembly101-720p re-encode is used
(720p AV1, 60 fps, same timing, 354 of 362 recordings); recordings it lacks
are listed in data/manifests/assembly101_missing.txt for a later
`--source official` run. Recordings with mistake annotations come first,
then the remaining ones with coarse annotations. Annotation frame numbers
are at 30 fps. Outputs: data/transcoded/assembly101/<view>/<recording>.mp4.
Safe to rerun: finished outputs are skipped.

    python scripts/fetch_assembly101.py [--source auto|official|mirror] [--limit N]
"""
import argparse
import re
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from disk_guard import wait_for_space  # noqa: E402
from fetch_common import Status, TranscodePool, setup_logging, wait_for_staging  # noqa: E402
from wcem.paths import ANNOTATIONS, MANIFESTS, RAW, TRANSCODED  # noqa: E402

OFFICIAL = "cvml-nus/assembly101"
MIRROR = "pablovela5620/assembly101-720p"
VIEWS = {"v4": ("C10119_rgb",), "e4": ("HMC_84358933_mono10bit", "HMC_21179183_mono10bit")}
MISTAKES = ANNOTATIONS / "assembly101_mistakes" / "annots"
HF_ANN = ANNOTATIONS / "assembly101_hf"
OUT = TRANSCODED / "assembly101"
STAGING = RAW / "assembly101"
STAGING_CAP_GB = 40


def repo_path(source, rec, cam):
    if source == OFFICIAL:
        return f"recordings/{rec}/{cam}.mp4"
    return f"videos/av1-720-new/{rec}/{cam}_low.mp4"


def has_access(repo):
    try:
        HfApi().auth_check(repo, repo_type="dataset")
        return True
    except Exception:
        return False


def plan(source, official_files, source_sizes):
    """Recordings in priority order, the (view, rec, repo path, size) items to fetch, and the gaps."""
    recs = {p.split("/")[1] for p in official_files if p.startswith("recordings/")}
    coarse = {m.group(1) for p in official_files
              if (m := re.match(r"annotations/coarse-annotations/coarse_labels/(?:dis)?assembly_(.+)\.txt$", p))}
    mistakes = {p.stem for p in MISTAKES.glob("*.csv")}
    order = sorted(mistakes & recs) + sorted((coarse - mistakes) & recs)
    items, missing = [], []
    for rec in order:
        for view, cams in VIEWS.items():
            rel = next((repo_path(source, rec, c) for c in cams if repo_path(source, rec, c) in source_sizes), None)
            if rel is None:
                on_official = any(f"recordings/{rec}/{c}.mp4" in official_files for c in cams)
                missing.append(f"{view}/{rec}\t{'only on ' + OFFICIAL if on_official else 'view not recorded'}")
            else:
                items.append((view, rec, rel, source_sizes[rel]))
    return order, items, missing, len(mistakes & recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["auto", "official", "mirror"], default="auto")
    ap.add_argument("--transcoders", type=int, default=2)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: only the first N items")
    args = ap.parse_args()

    log = setup_logging("assembly101")
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    official_files = {f.rfilename: f.size for f in api.dataset_info(OFFICIAL, files_metadata=True).siblings}
    access = has_access(OFFICIAL)
    source = OFFICIAL if args.source == "official" or (args.source == "auto" and access) else MIRROR
    if access:
        snapshot_download(OFFICIAL, repo_type="dataset", allow_patterns=["annotations/**"], local_dir=HF_ANN)
    else:
        log.warning("no access to gated %s (set HF_TOKEN after accepting its terms); annotations not fetched",
                    OFFICIAL)
    sizes = official_files if source == OFFICIAL else {
        f.rfilename: f.size for f in api.dataset_info(MIRROR, files_metadata=True).siblings}
    order, items, missing, n_mistake = plan(source, official_files, sizes)
    (MANIFESTS / "assembly101.txt").write_text("\n".join(order) + "\n")
    (MANIFESTS / "assembly101_missing.txt").write_text("".join(m + "\n" for m in missing))
    todo = [it for it in items if not (OUT / it[0] / f"{it[1]}.mp4").exists()][: args.limit]
    log.info("source %s: %d recordings (%d with mistake labels), %d view files (%.0f GB raw), %d views missing, "
             "%d to do", source, len(order), n_mistake, len(items), sum(i[3] for i in items) / 1e9, len(missing),
             len(todo))

    status = Status(MANIFESTS / "assembly101_status.csv")
    pool = TranscodePool(status, log, workers=args.transcoders, threads=args.threads, tol_s=1.0)
    for view, rec, rel, size in todo:
        dst = OUT / view / f"{rec}.mp4"
        wait_for_staging(STAGING, STAGING_CAP_GB, log)
        wait_for_space(size / 1e9 + 5, log=log.info)
        t0 = time.time()
        path = None
        for attempt in range(3):
            try:
                path = Path(hf_hub_download(source, rel, repo_type="dataset", local_dir=STAGING / "hf"))
            except Exception as e:
                log.warning("download %s failed (attempt %d): %s", rel, attempt + 1, str(e)[-200:])
                time.sleep(120 * (attempt + 1))
                continue
            if path.stat().st_size == size:
                break
            log.warning("size mismatch %s (%d != %d); discarding", rel, path.stat().st_size, size)
            path.unlink()
            path = None
        if path is None:
            log.error("download failed: %s", rel)
            status.write(key=f"{view}/{rec}", status="download_failed")
            continue
        log.info("downloaded %s: %.2f GB in %.0fs (%.1f MB/s)", rel, size / 1e9, time.time() - t0,
                 size / 1e6 / max(time.time() - t0, 1))
        pool.put(f"{view}/{rec}", path, dst)
    pool.close()
    log.info("done; failed: %s", pool.failed or "none")


if __name__ == "__main__":
    main()
