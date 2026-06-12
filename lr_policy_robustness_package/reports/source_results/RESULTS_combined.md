# Combined Results — Night10h + Catchup

Generated: 2026-05-13T12:20:11

All seeds across both nights merged into a single per-variant view.
Completed = ran the full 35 epochs.  Partial runs are listed but not
included in the headline mean/std.

**Reference cosine baseline**: 69.81 ± 0.33 pp (4 prior seeds, depth_list=110_20, ratio=4, batch=256, 35 epochs).

## Headline (completed seeds only)

| Variant | n | seeds | mean ± std | Δ vs cosine 69.81 | best peak | best variant |
|---|---:|---|---|---:|---:|---|
| `cosine` | 2 | [10, 11] | **70.02** ± 0.01 | +0.21 | 70.03 | seed=11 |
| `curve_veto_clamp07` | 1 | [10] | **50.95** ± 0.00 | -18.86 | 64.09 | seed=10 |
| `hypergrad` | 4 | [1, 2, 3, 10] | **45.80** ± 4.99 | -24.01 | 54.94 | seed=10 |
| `residual_log` | 6 | [1, 2, 3, 10, 11, 12] | **69.31** ± 0.43 | -0.50 | 69.91 | seed=1 |
| `sf_adamw` | 3 | [1, 2, 3] | **49.37** ± 0.71 | -20.44 | 50.05 | seed=3 |
| `sf_sgd` | 4 | [1, 2, 3, 10] | **52.61** ± 1.67 | -17.20 | 55.10 | seed=2 |

## Per-run breakdown (incl. partial)

| Variant | Seed | Source | Epochs | Status | Final acc | Best acc | Best epoch |
|---|---:|---|---:|---|---:|---:|---:|
| `cosine` | 10 | industry-night | 37 | OK | 70.01 | 70.02 | 34 |
| `cosine` | 11 | industry-night | 36 | OK | 70.03 | 70.03 | 35 |
| `curve_veto_clamp07` | 10 | industry-night | 36 | OK | 50.95 | 64.09 | 25 |
| `hypergrad` | 1 | stars-night | 36 | OK | 43.66 | 53.50 | 31 |
| `hypergrad` | 2 | stars-catchup | 36 | OK | 47.26 | 50.55 | 13 |
| `hypergrad` | 3 | stars-catchup | 36 | OK | 51.95 | 53.89 | 18 |
| `hypergrad` | 10 | industry-night | 36 | OK | 40.32 | 54.94 | 18 |
| `residual_log` | 1 | stars-catchup | 36 | OK | 69.91 | 69.91 | 35 |
| `residual_log` | 2 | stars-catchup | 36 | OK | 69.01 | 69.01 | 35 |
| `residual_log` | 3 | stars-catchup | 36 | OK | 69.64 | 69.64 | 35 |
| `residual_log` | 10 | industry-night | 36 | OK | 68.79 | 68.99 | 34 |
| `residual_log` | 11 | industry-night | 36 | OK | 69.45 | 69.45 | 35 |
| `residual_log` | 12 | industry-catchup | 36 | OK | 69.03 | 69.04 | 34 |
| `residual_log` | 13 | industry-catchup | 9 | partial | 48.63 | 48.63 | 8 |
| `sf_adamw` | 1 | stars-night | 36 | OK | 48.64 | 48.64 | 35 |
| `sf_adamw` | 2 | stars-catchup | 36 | OK | 49.42 | 49.42 | 35 |
| `sf_adamw` | 3 | stars-catchup | 36 | OK | 50.05 | 50.05 | 35 |
| `sf_sgd` | 1 | stars-night | 36 | OK | 51.87 | 53.29 | 22 |
| `sf_sgd` | 2 | stars-catchup | 36 | OK | 55.10 | 55.10 | 35 |
| `sf_sgd` | 3 | stars-catchup | 36 | OK | 51.48 | 52.18 | 30 |
| `sf_sgd` | 10 | industry-night | 36 | OK | 52.01 | 54.17 | 34 |

## Verdict

### Top-3 by mean final accuracy

1. `cosine` — **70.02 ± 0.01** pp (n=2, peak=70.03, Δ vs cosine = +0.21)
2. `residual_log` — **69.31 ± 0.43** pp (n=6, peak=69.91, Δ vs cosine = -0.50)
3. `sf_sgd` — **52.61 ± 1.67** pp (n=4, peak=55.10, Δ vs cosine = -17.20)

### Main finding

**P3 residual-on-log-cosine head** (`--policy_output residual_log`) reached
**69.31 ± 0.43 pp** across **6 seeds** vs cosine 69.81 ± 0.33.

Gap to cosine: **Δ = -0.50 pp** (peak run = 69.91).

95% CI on residual_log mean: [68.96, 69.65] pp.
Cosine 95% CI (n=4): [69.49, 70.13] pp.

**SLIGHTLY BELOW COSINE** but well within tight learnable-policy regime.

## Per-variant detailed seed list

### `cosine` (n=2)

- seed 10 (industry-night): 70.01
- seed 11 (industry-night): 70.03

### `curve_veto_clamp07` (n=1)

- seed 10 (industry-night): 50.95

### `hypergrad` (n=4)

- seed 1 (stars-night): 43.66
- seed 2 (stars-catchup): 47.26
- seed 3 (stars-catchup): 51.95
- seed 10 (industry-night): 40.32

### `residual_log` (n=6)

- seed 1 (stars-catchup): 69.91
- seed 2 (stars-catchup): 69.01
- seed 3 (stars-catchup): 69.64
- seed 10 (industry-night): 68.79
- seed 11 (industry-night): 69.45
- seed 12 (industry-catchup): 69.03

### `sf_adamw` (n=3)

- seed 1 (stars-night): 48.64
- seed 2 (stars-catchup): 49.42
- seed 3 (stars-catchup): 50.05

### `sf_sgd` (n=4)

- seed 1 (stars-night): 51.87
- seed 2 (stars-catchup): 55.10
- seed 3 (stars-catchup): 51.48
- seed 10 (industry-night): 52.01
