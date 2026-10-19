# Stage B

Stage A anchor: `checkpoints/stage_a_decoder_tune/step4000.pt` (Kalman σ_obs=0.05 + decoder tune).

**Pipeline (EK100, in order):**

1. `configs/stage_b.yaml` — L pretrain (`slow_only`, `slow_read: false`)
2. `configs/stage_b_joint.yaml` — fusion + gates/decoders (`slow_read: true`, `precision_fusion: true`)
3. `configs/stage_b_joint_readout.yaml` — readout calibration (`joint_readout_tune`, `rate_weight_max: 0`)

**Ship checkpoint:** `checkpoints/stage_b_joint_readout/step1000.pt` (run Gate 2 with `run_gate2_stage_b.sh`).

Historical note: Stage B was quarantined until fast-memory E4 passed on the Kalman path.

Reason: E4 showed **higher** write probability on near-duplicate steps; E1 on Stage B checkpoints showed **negative memory gain** vs swapped `S`, consistent with competitive interference from `L` rather than a broken episodic core.

Do **not** run `configs/stage_b.yaml` until [GATE_CRITERIA.md](GATE_CRITERIA.md) is satisfied:

1. **E1** — positive gain all lag bins (val + train). **Met** on `step13000`.
2. **E2** — `beats_uniform_all` on the eval λ grid. **Not met** (see `checkpoints/gate1/e2_sweep_*.json`).
3. **E4** — `repeat_lt_first` on train split. **Not met** (see `checkpoints/gate2/e4_*`).
4. Use precision fusion for any `L` training ([STAGE_L_PRETRAIN.md](STAGE_L_PRETRAIN.md)); no blind `v - L(k)` subtractor.

Anchor checkpoint: `checkpoints/stage_a/step13000.pt` (frozen gates).

Optional fast-memory pivot: `kalman_gate` / Stage A′ — see `configs/stage_a_kalman.yaml` after M5.5 passes (`scripts/synthetic_m5_kalman.py`).
