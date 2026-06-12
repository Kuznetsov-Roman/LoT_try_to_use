# SOTA Trainable LR Comparison

Generated: 2026-05-15 17:20:57 UTC

Baselines: cosine seed3 = 70.45; PatchTST headline = 70.70; cosine 3-seed mean = 70.197.

## Verdict
No parsed run beats cosine seed3 yet. Best run `adalrs_default_seed1_60ep` reached 68.22.

## Method Summary

| method | variant | n | final mean | final std | last10 mean | delta vs cosine mean |
|---|---|---:|---:|---:|---:|---:|
| adalrs | `default` | 3 | 67.410 | 0.820 | 66.356 | -2.787 |
| adalrs | `aggressive` | 3 | 65.990 | 2.287 | 65.521 | -4.207 |
| adalrs | `narrow_safe` | 3 | 54.733 | 4.924 | 50.275 | -15.464 |
| bandit_exp3 | `safe` | 3 | 53.613 | 17.941 | 48.370 | -16.584 |
| bandit_exp3 | `fast` | 3 | 52.093 | 6.493 | 44.698 | -18.104 |
| hypergrad_hb | `fast` | 3 | 42.670 | 8.209 | 41.350 | -27.527 |
| bandit_ucb | `ucb` | 3 | 40.883 | 23.435 | 40.596 | -29.314 |
| hypergrad_hb | `safe` | 3 | 39.747 | 3.950 | 42.586 | -30.450 |
| hypergrad_hb | `smooth` | 3 | 34.090 | 8.059 | 40.166 | -36.107 |

## Top Runs

| rank | server | run | final acc | final loss | best acc | last10 | vs cosine seed3 | vs PatchTST |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | stars | `adalrs_default_seed1_60ep` | 68.22 | 1.152 | 68.91 | 66.79 | -2.23 | -2.48 |
| 2 | stars | `adalrs_aggressive_seed3_60ep` | 67.83 | 1.122 | 69.35 | 66.59 | -2.62 | -2.87 |
| 3 | stars | `adalrs_default_seed2_60ep` | 67.43 | 1.157 | 69.19 | 66.30 | -3.02 | -3.27 |
| 4 | stars | `adalrs_aggressive_seed2_60ep` | 66.71 | 1.177 | 69.20 | 65.40 | -3.74 | -3.99 |
| 5 | stars | `adalrs_default_seed3_60ep` | 66.58 | 1.181 | 69.18 | 65.98 | -3.87 | -4.12 |
| 6 | industry | `bandit_ucb_seed1_60ep` | 66.58 | 1.165 | 67.66 | 53.97 | -3.87 | -4.12 |
| 7 | industry | `bandit_exp3_safe_seed3_60ep` | 64.61 | 1.245 | 65.57 | 53.68 | -5.84 | -6.09 |
| 8 | stars | `adalrs_aggressive_seed1_60ep` | 63.43 | 1.288 | 68.44 | 64.57 | -7.02 | -7.27 |
| 9 | industry | `bandit_exp3_safe_seed2_60ep` | 63.32 | 1.278 | 66.89 | 44.39 | -7.13 | -7.38 |
| 10 | industry | `bandit_exp3_fast_seed2_60ep` | 57.86 | 1.493 | 66.23 | 46.98 | -12.59 | -12.84 |
| 11 | stars | `adalrs_narrow_safe_seed1_60ep` | 57.73 | 1.491 | 67.15 | 58.06 | -12.72 | -12.97 |
| 12 | stars | `adalrs_narrow_safe_seed2_60ep` | 57.42 | 1.505 | 63.52 | 49.84 | -13.03 | -13.28 |
| 13 | industry | `bandit_exp3_fast_seed3_60ep` | 53.36 | 1.636 | 66.05 | 38.33 | -17.09 | -17.34 |
| 14 | stars | `adalrs_narrow_safe_seed3_60ep` | 49.05 | 1.855 | 63.02 | 42.93 | -21.40 | -21.65 |
| 15 | mlspace | `hypergrad_hb_fast_seed3_60ep` | 47.77 | 1.991 | 52.24 | 41.86 | -22.68 | -22.93 |
| 16 | mlspace | `hypergrad_hb_fast_seed2_60ep` | 47.04 | 2.081 | 53.28 | 43.73 | -23.41 | -23.66 |
| 17 | industry | `bandit_exp3_fast_seed1_60ep` | 45.06 | 2.156 | 64.00 | 48.79 | -25.39 | -25.64 |
| 18 | mlspace | `hypergrad_hb_safe_seed3_60ep` | 44.26 | 2.243 | 52.34 | 43.96 | -26.19 | -26.44 |
| 19 | mlspace | `hypergrad_hb_smooth_seed2_60ep` | 39.01 | 2.526 | 52.65 | 41.94 | -31.44 | -31.69 |
| 20 | mlspace | `hypergrad_hb_smooth_seed3_60ep` | 38.47 | 2.626 | 52.67 | 39.48 | -31.98 | -32.23 |
