"""Write EK100 download manifests in milestone order.

d2: ~10 h of train videos ranked by washing time (Stage 0, E0), plus ~1 h of
    validation videos with washing, held out for the predictor checks.
d3: every remaining video, train first, then validation, then test.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wcem.paths import ANNOTATIONS, MANIFESTS  # noqa: E402

EK = ANNOTATIONS / "ek100"
TRAIN_HOURS = 10.0
VAL_HOURS = 1.0
MAX_PER_PARTICIPANT_H = 2.0


def seconds(ts):
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def wash_seconds(df):
    df = df[df.verb == "wash"]
    dur = df.stop_timestamp.map(seconds) - df.start_timestamp.map(seconds)
    return dur.groupby(df.video_id).sum()


def pick(wash, info, budget_h):
    # rank by washing seconds weighted by washing fraction, cap hours per participant for diversity
    cand = pd.DataFrame({"wash": wash, "dur": info.duration.reindex(wash.index)})
    cand["score"] = cand.wash * (cand.wash / cand.dur).clip(upper=0.5)
    chosen, total, per_p = [], 0.0, {}
    for vid, row in cand.sort_values("score", ascending=False).iterrows():
        p = vid.split("_")[0]
        h = row.dur / 3600
        if per_p.get(p, 0.0) + h > MAX_PER_PARTICIPANT_H or total + h > budget_h:
            continue
        chosen.append(vid)
        per_p[p] = per_p.get(p, 0.0) + h
        total += h
    return chosen, total, cand.loc[chosen, "wash"].sum() / 3600


def main():
    train = pd.read_csv(EK / "EPIC_100_train.csv")
    val = pd.read_csv(EK / "EPIC_100_validation.csv")
    info = pd.read_csv(EK / "EPIC_100_video_info.csv").set_index("video_id")
    train_ids, val_ids = set(train.video_id), set(val.video_id)

    d2_train, h_tr, w_tr = pick(wash_seconds(train), info, TRAIN_HOURS)
    d2_val, h_va, w_va = pick(wash_seconds(val), info, VAL_HOURS)
    d2 = d2_train + d2_val

    def split(v):
        return 0 if v in train_ids else 1 if v in val_ids else 2

    rest = [v for v in info.index if v not in set(d2)]
    rest.sort(key=lambda v: (split(v), v))

    MANIFESTS.mkdir(parents=True, exist_ok=True)
    (MANIFESTS / "ek100_d2_train.txt").write_text("\n".join(d2_train) + "\n")
    (MANIFESTS / "ek100_d2_val.txt").write_text("\n".join(d2_val) + "\n")
    (MANIFESTS / "ek100_d2.txt").write_text("\n".join(d2) + "\n")
    (MANIFESTS / "ek100_d3.txt").write_text("\n".join(rest) + "\n")

    print(f"d2 train: {len(d2_train)} videos, {h_tr:.2f} h, washing {w_tr:.2f} h, "
          f"participants {sorted({v.split('_')[0] for v in d2_train})}")
    print(f"d2 val:   {len(d2_val)} videos, {h_va:.2f} h, washing {w_va:.2f} h")
    print(f"d3: {len(rest)} videos, {info.duration.reindex(rest).sum() / 3600:.1f} h")


if __name__ == "__main__":
    main()
