# Slow memory L — pretrain and Stage B entry

## Prerequisites

1. Gate 1 E1 pass (anchor: `step13000`).
2. Gate 1 E2 `beats_uniform_all` (or documented relaxed band).
3. Gate 2 E4 `repeat_lt_first` on fast memory **without** blind subtractor failures.

## Tier 1 — precision fusion (before joint Stage B)

- Enable `precision_fusion: true` with `slow_enabled: true` in config.
- `SlowMemory.forward_with_variance` supplies σ_L² from ensemble spread; [`wcem/memory/precision.py`](../wcem/memory/precision.py) fuses L and S predictions instead of `v' = v - L(k)`.
- **Do not** use raw `v - L` in Stage B configs.

## L pretrain objective (proposed)

1. Freeze representation from `step13000` (tokenizer, window, keys, cues).
2. Train only `SlowMemory` P/U to minimize `|v - μ_L(k)|` on **repeat** steps (mask from `_dupsim ≥ 0.85`) while keeping ensemble disagreement calibrated on held-out videos.
3. Enable precision fusion; run E4 — `repeat_minus_first` should improve vs blind subtract.

## Stage B joint training

- Load L pretrain + fast gate checkpoint.
- Train gates/decoders/L with precision fusion on; compare E1/E2/E4 after each milestone.
- First `slow_enabled` E4 must be diffed against fast-only E4 at **matched** `writes_per_step`.
