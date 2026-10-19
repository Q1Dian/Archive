# Gate 1 and Gate 2 exit criteria

Use the harness in `scripts/eval_e1.py`, `eval_e2.py`, `eval_e4.py`. **Stage A anchor:** `checkpoints/stage_a_decoder_tune/step4000.pt`.

## Gate 1 — fast memory recall and selective economics

| Check | Criterion | Status (Oct 18) |
|-------|-----------|-----------------|
| **E1** | Trained memory gain **positive in every lag bin** (8–16 … 256+) on **val** and **train**, above FIFO byte-cap baseline | **Pass** on `step13000` (`checkpoints/gate1/e1_step13000_{val,train}.json`) |
| **E2** | At eval λ grid, trained selective writes **beat uniform** at **matched write count** (`beats_uniform_all`) | **Fail** (3/6 frozen, 4/6 gate_tune); see `checkpoints/gate1/e2_*.json` |

## Gate 2 — duplicate thrift (fast memory, `L` disabled)

| Check | Criterion | Status (Oct 18) |
|-------|-----------|-----------------|
| **E4** | Aggregate `p_write_repeat` **<** `p_write_first` (`repeat_lt_first == true`) on **train** split, sim ≥ 0.85 | **Fail** (frozen saturated; gate_tune 1.74× more on repeats) |

## Stage B (`slow_enabled`)

Do **not** start until **both** Gate 1 E2 and Gate 2 E4 pass on a **fast-only** checkpoint at the chosen deployment `log_lam`. See [STAGE_B_QUARANTINE.md](STAGE_B_QUARANTINE.md).

## λ sweep baselines

Run `scripts/gate_lam_sweep.py` before architecture pivots. Summary: `checkpoints/gate1/lam_sweep_summary.json`.

**Oct 18 sweep:** No λ achieved `repeat_lt_first` on PairGate (`e4_any_repeat_lt_first: false` for `step13000` and `gate_tune`). E2 best: gate_tune **5/7** λ beat uniform (still not `beats_uniform_all`).

## Stage A′ Kalman ablation (`configs/stage_a_kalman.yaml`)

After M5.5 pass (`scripts/synthetic_m5_kalman.py`): 1500-step train from frozen `step13000` repr.

| Metric | `step1500` kalman | PairGate gate_tune @ -4.6 |
|--------|-------------------|----------------------------|
| E4 `repeat_lt_first` | **true** (−1.9 pp repeat vs first) | false (+7.8 pp) |
| E4 frac videos pass | **72%** | ~10% |
| E1 val 32–64 gain | **−2.6** (regressed) | +10.0 |

Kalman gate improves Gate 2 thrift; recall (E1) still needs longer train or hybrid policy. Artifacts: `checkpoints/gate2/e4_kalman_train.json`, `checkpoints/gate1/e1_kalman_val.json`, `e2_kalman.json`.

**Oct 18 extended run (`step8000`, `decay_on_write_only: true`):** `checkpoints/stage_a_kalman/step8000.pt`

| Eval | Result |
|------|--------|
| E1 train | **Pass** all bins (+10–14) |
| E1 val | **Fail** 32–64 (−5.2), 64–128 (−35.7); 3 videos noisy |
| E4 | **Pass** `repeat_lt_first` (−1.8 pp), 72% videos |
| E2 | **Fail** 2/6 λ beat uniform |

Log: `data/logs/stage_a_kalman_8k.out`

**Full D2 val (Oct 18):** Manifest has **8** videos; only **3** have ≥512 cached steps (`P01_15`, `P13_01`, `P27_05`). Multi-window eval uses `--window-stride 128` → **6 windows** (`scripts/run_full_val_e1_e2.sh`).

| Eval @ `step8000` | Full val (6 windows) |
|-------------------|----------------------|
| E2 | **Pass** `beats_uniform_all` (6/6 λ) — `e2_step8000_val_full.json` |
| E1 | **Mixed** — 32–64 **−2.2**, 64–128 **−11.5**; 128–256 **+3.9**, 256+ **+6.1** — `e1_step8000_val_full.json` |

**`sigma_obs` sweep (eval-only, 11 values 0.05–1.0 + spot 0.01–0.03):** `checkpoints/gate1/sigma_obs_sweep/summary.json`

- **E4:** `repeat_lt_first` **true** at every σ (duplicate thrift robust).
- **E1 val 32–64 / 64–128:** stays **negative** across σ; best 32–64 ≈ **−1.1** at σ=0.01 (still fails). **No eval-only calibration unlocks val E1.**
- Script: `scripts/sweep_sigma_obs.py`; use `--sigma-obs` on `eval_e1.py` / `eval_e4.py`.
