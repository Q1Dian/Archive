"""Kinetics-GEBD val: Kinetics-400 val shards -> keep GEBD clips -> 8 fps transcode.

Shards come from the Hugging Face copy of the CVDF tarballs
(kiyoonkim/kinetics-400-targz, byte-identical sizes; ~10x faster than S3),
with the CVDF S3 bucket as fallback. Only clips listed in
k400_val_raw_annotation.pkl are extracted; they are transcoded to 8 fps /
384 px short side (so 4 fps is exact every-other-frame subsampling) and the
shard and raw clips are deleted. Clips missing from the shards or unreadable
are looked up in the CVDF replacement archive before being recorded in
data/manifests/gebd_val_missing.txt. Safe to rerun.

    python scripts/fetch_gebd.py [--limit-shards N] [--limit-clips N]
"""
import argparse
import json
import pickle
import sys
import tarfile
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from disk_guard import wait_for_space  # noqa: E402
from fetch_common import Status, TranscodePool, http_download, setup_logging, wait_for_staging  # noqa: E402
from wcem.paths import ANNOTATIONS, MANIFESTS, RAW, TRANSCODED  # noqa: E402

ANN = ANNOTATIONS / "gebd" / "k400_val_raw_annotation.pkl"
HF_REPO = "kiyoonkim/kinetics-400-targz"
S3 = "https://s3.amazonaws.com/kinetics/400/"
SHARDS = [f"val/part_{i}.tar.gz" for i in range(20)]
REPLACEMENT = "replacement/replacement_for_corrupted_k400.tgz"
S3_PATHS = {REPLACEMENT: "replacement_for_corrupted_k400.tgz"}
OUT = TRANSCODED / "gebd"
STAGING = RAW / "gebd"
STAGING_CAP_GB = 15
FPS = 8


def fetch_shard(rel, size, log):
    """Download one tarball to staging, verified by size; HF first, S3 fallback."""
    for attempt in range(3):
        t0 = time.time()
        try:
            path = Path(hf_hub_download(HF_REPO, rel, repo_type="dataset", local_dir=STAGING / "hf"))
        except Exception as e:
            log.warning("hf download %s failed (attempt %d): %s", rel, attempt + 1, str(e)[-200:])
            time.sleep(60 * (attempt + 1))
            continue
        if path.stat().st_size == size:
            log.info("downloaded %s from hf: %.2f GB in %.0fs (%.1f MB/s)", rel, size / 1e9, time.time() - t0,
                     size / 1e6 / max(time.time() - t0, 1))
            return path
        log.warning("size mismatch %s from hf (%d != %d); discarding", rel, path.stat().st_size, size)
        path.unlink()
    t0 = time.time()
    path = http_download(S3 + S3_PATHS.get(rel, rel), STAGING / "s3" / rel, log)
    log.info("downloaded %s from s3: %.2f GB in %.0fs", rel, path.stat().st_size / 1e9, time.time() - t0)
    return path


def extract(tgz, wanted, pool, log, limit=None):
    """Extract members whose basename is in `wanted` (basename -> clip id) and queue them for transcoding."""
    found = []
    t0 = time.time()
    with tarfile.open(tgz, mode="r|gz") as tar:
        for m in tar:
            name = Path(m.name).name
            if not m.isfile() or name not in wanted:
                continue
            cid = wanted[name]
            src = STAGING / name
            found.append(cid)
            if (OUT / f"{cid}.mp4").exists() or src.exists():  # done, or already queued
                continue
            part = src.with_name(src.name + ".part")
            with tar.extractfile(m) as fin:
                part.write_bytes(fin.read())
            part.rename(src)
            pool.put(cid, src, OUT / f"{cid}.mp4")
            if limit and len(found) >= limit:
                break
    log.info("extracted %d GEBD clips from %s in %.0fs", len(found), tgz.name, time.time() - t0)
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcoders", type=int, default=6)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit-shards", type=int, default=None, help="smoke test: only the last N shards")
    ap.add_argument("--limit-clips", type=int, default=None, help="smoke test: stop after N clips of a shard")
    ap.add_argument("--no-replacement", action="store_true")
    args = ap.parse_args()
    smoke = args.limit_shards is not None or args.limit_clips is not None

    log = setup_logging("gebd")
    STAGING.mkdir(parents=True, exist_ok=True)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    ann = pickle.load(open(ANN, "rb"))
    clip_ids = list(ann)
    (MANIFESTS / "gebd_val.txt").write_text("\n".join(clip_ids) + "\n")
    by_name = {Path(v["path_video"]).name: k for k, v in ann.items()}
    todo = {n: c for n, c in by_name.items() if not (OUT / f"{c}.mp4").exists()}
    log.info("%d GEBD val clips, %d to do", len(by_name), len(todo))
    sizes = {f.rfilename: f.size for f in HfApi().dataset_info(HF_REPO, files_metadata=True).siblings}

    status = Status(MANIFESTS / "gebd_val_status.csv")
    pool = TranscodePool(status, log, workers=args.transcoders, threads=args.threads, delete_bad=True,
                         tol_s=0.5, fps=FPS)
    for src in STAGING.glob("*.mp4"):  # left over from an interrupted run
        if src.name in todo:
            pool.put(todo[src.name], src, OUT / f"{todo[src.name]}.mp4")

    done_path = STAGING / "shards_done.json"
    done = json.loads(done_path.read_text()) if done_path.exists() else {}

    def process(rel, wanted, limit=None):
        wait_for_staging(STAGING, STAGING_CAP_GB - sizes[rel] / 1e9, log)
        wait_for_space(sizes[rel] / 1e9 + 2, log=log.info)
        try:
            tgz = fetch_shard(rel, sizes[rel], log)
            found = extract(tgz, wanted, pool, log, limit)
        except Exception as e:
            log.error("shard %s failed: %s", rel, str(e)[-300:])
            return
        if limit:
            return
        tgz.unlink()
        done[rel] = found
        done_path.write_text(json.dumps(done))

    shards = SHARDS[-args.limit_shards:] if args.limit_shards else SHARDS
    for rel in shards:
        if rel not in done and todo:
            process(rel, todo, args.limit_clips)
    while pool.pending():
        time.sleep(5)

    if not smoke and not args.no_replacement and REPLACEMENT not in done:
        in_shards = {c for v in done.values() for c in v}
        rows = status.read()
        lookup = {n: c for n, c in by_name.items()
                  if c not in in_shards or rows.get(c, {}).get("status") == "corrupt"}
        log.info("%d clips missing from the val shards or corrupt; checking the replacement archive", len(lookup))
        if lookup and len(done) == len(SHARDS):
            process(REPLACEMENT, lookup)
    pool.close()

    if smoke:
        log.info("smoke test done; failed: %s", pool.failed or "none")
        return
    rows = status.read()
    bad = []
    for c in clip_ids:
        if not (OUT / f"{c}.mp4").exists():
            bad.append(f"{c}\t{rows.get(c, {}).get('status', 'missing')}")
        elif (rows.get(c, {}).get("status") == "suspect"
              and abs(float(rows[c]["out_s"]) - ann[c]["video_duration"]) > 2 / FPS):
            bad.append(f"{c}\tsuspect: {rows[c]['out_s']}s vs annotated {ann[c]['video_duration']:.2f}s")
    (MANIFESTS / "gebd_val_missing.txt").write_text("".join(b + "\n" for b in bad))
    log.info("done: %d/%d clips transcoded, %d missing/corrupt/suspect (see gebd_val_missing.txt)",
             sum((OUT / f"{c}.mp4").exists() for c in clip_ids), len(clip_ids), len(bad))


if __name__ == "__main__":
    main()
