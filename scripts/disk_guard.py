"""Free-space check shared by every download job.

Usage as a script prints the current state; jobs import `wait_for_space`.
"""
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wcem.paths import DATA  # noqa: E402

MIN_FREE_GB = 150
PROJECT_BUDGET_GB = 1500


def free_gb(path=DATA):
    return shutil.disk_usage(path).free / 1e9


def used_gb(path=DATA):
    total = 0
    for p in Path(path).rglob("*"):
        if p.is_file() and not p.is_symlink():
            total += p.stat().st_size
    return total / 1e9


def has_space(need_gb=0.0):
    return free_gb() - need_gb >= MIN_FREE_GB


def wait_for_space(need_gb=0.0, poll_s=300, log=print):
    while not has_space(need_gb):
        log(f"disk guard: {free_gb():.0f} GB free, need {need_gb:.0f} GB + {MIN_FREE_GB} GB floor")
        time.sleep(poll_s)


if __name__ == "__main__":
    DATA.mkdir(parents=True, exist_ok=True)
    print(f"free: {free_gb():.0f} GB  data/: {used_gb():.1f} GB  "
          f"floor: {MIN_FREE_GB} GB  budget: {PROJECT_BUDGET_GB} GB")
