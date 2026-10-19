import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("WCEM_DATA", REPO / "data"))
ANNOTATIONS = DATA / "annotations"
RAW = DATA / "raw"
TRANSCODED = DATA / "transcoded"
CACHE = DATA / "cache"
MANIFESTS = DATA / "manifests"
LOGS = DATA / "logs"
CHECKPOINTS = REPO / "checkpoints"
THIRD_PARTY = REPO / "third_party"
VJEPA2 = THIRD_PARTY / "vjepa2"
FFMPEG = REPO / ".conda" / "bin" / "ffmpeg"
