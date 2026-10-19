"""Assembly101: transcodes live under <view>/<recording>.mp4; cache keys use view__recording."""
from pathlib import Path

from wcem.paths import MANIFESTS, TRANSCODED

VIEW_ORDER = ("e4", "v4")


def cache_id(view: str, recording_id: str) -> str:
    return f"{view}__{recording_id}"


def parse_recording_id(entry: str):
    entry = entry.strip()
    if not entry:
        return None, None
    if "/" in entry:
        view, rec = entry.split("/", 1)
        return view, rec
    return None, entry


def resolve_transcode(root: Path, entry: str, prefer_view: str | None = None):
    """Return (mp4_path, cache_vid) or (None, None)."""
    view, rec = parse_recording_id(entry)
    if view:
        p = root / view / f"{rec}.mp4"
        return (p, cache_id(view, rec)) if p.exists() else (None, None)
    order = [prefer_view] if prefer_view else list(VIEW_ORDER)
    for v in order:
        if v is None:
            continue
        p = root / v / f"{rec}.mp4"
        if p.exists():
            return p, cache_id(v, rec)
    return None, None


def manifest_transcodes(manifest_path: Path, prefer_view: str | None = "e4"):
    root = TRANSCODED / "assembly101"
    out = {}
    for line in manifest_path.read_text().splitlines():
        p, cid = resolve_transcode(root, line, prefer_view)
        if p is not None:
            out[cid] = p
    return out


def manifest_cache_ids(manifest_path: Path | None = None, prefer_view: str | None = "e4"):
    mf = manifest_path or (MANIFESTS / "assembly101.txt")
    return list(manifest_transcodes(mf, prefer_view).keys())
