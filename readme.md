# Write-cost episodic memory on V-JEPA 2.1

Streaming event segmentation and episodic memory on a frozen V-JEPA 2.1 encoder,
trained on coding length (write cost plus recall error). Design:
[full plan](https://cursor.com/canvas/store-60347d68-5030-4353-a548-49bff245fe0d/5336ddef-2768-46cd-9bd1-77d606a73a9e).

## Layout

```
wcem/data/        download queue, transcoding, clip timeline, cached-feature sampler, label alignment
wcem/perception/  V-JEPA 2.1 wrapper, Stage 0 predictor, extraction
wcem/models/      slots, window, projections, decoders, slow memory L
wcem/memory/      recurrence for S, hard-concrete gate, event log
wcem/eval/        evaluation harnesses and baselines
scripts/          command-line entry points
configs/          yaml per stage
tests/
third_party/      external code (vjepa2), not tracked
data/             videos, transcodes, feature caches, not tracked
```

## Environment

```
conda env create -p ./.conda -f environment.yml
conda activate ./.conda
git clone --depth 1 https://github.com/facebookresearch/vjepa2.git third_party/vjepa2
pytest tests
```

`requirements-optional.txt` lists packages used only by ablations, baselines and the Ego4D download.

## Data

All datasets live under `data/` and are fetched in the order the milestones need them:
EK100 annotations and a washing-heavy subset, the rest of EK100, Kinetics-GEBD val,
Ego4D GoalStep (needs a signed license), Assembly101, CaptainCook4D.
Raw videos are transcoded to 4 fps at a 384 px short side and deleted.
`scripts/disk_guard.py` pauses downloads when free space drops below 150 GB.
