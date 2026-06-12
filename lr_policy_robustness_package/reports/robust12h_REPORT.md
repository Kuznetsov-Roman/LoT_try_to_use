# Robustness Campaign — 12h × 3 servers

Cosine reference (35ep, 4 prior seeds): **69.81 ± 0.33**
Cosine reference (60ep, 3 prior seeds): **70.2 ± 0.23**
PatchTST single-seed headline (60ep): **70.7**

Total runs scanned: **167** (completed: 167, partial/failed: 0)

## 1. Per-cell summary (mean ± std over seeds)

| server | experiment | method | perturb | n | final_mean | final_std | last10_mean | Δ vs cos ref | shock_dip | recov_ep |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| industry | compound | cosine | label=0.20+shock_lr=1.0@e15 | 4 | 64.08 | 0.35 | 57.84 | -5.73 | — | — |
| industry | compound | patchtst | label=0.20+shock_lr=1.0@e15 | 4 | 63.11 | 0.44 | 58.83 | -6.70 | — | — |
| industry | compound | residgru | label=0.20+shock_lr=1.0@e15 | 3 | 62.30 | 0.35 | 60.10 | -7.51 | — | — |
| industry | innoise | cosine | input_noise=0.05 | 4 | 65.98 | 0.43 | 62.20 | -3.83 | — | — |
| industry | innoise | patchtst | input_noise=0.05 | 4 | 65.90 | 0.44 | 61.94 | -3.91 | — | — |
| industry | innoise | cosine | input_noise=0.15 | 4 | 57.33 | 0.37 | 53.84 | -12.48 | — | — |
| industry | innoise | patchtst | input_noise=0.15 | 4 | 56.99 | 0.11 | 53.45 | -12.82 | — | — |
| industry | lblnoise | cosine | label_noise=0.10 | 4 | 66.60 | 0.49 | 62.48 | -3.21 | — | — |
| industry | lblnoise | patchtst | label_noise=0.10 | 4 | 66.34 | 0.23 | 62.02 | -3.47 | — | — |
| industry | lblnoise | residgru | label_noise=0.10 | 4 | 65.45 | 0.61 | 61.35 | -4.36 | — | — |
| industry | lblnoise | cosine | label_noise=0.20 | 4 | 63.71 | 0.23 | 58.94 | -6.10 | — | — |
| industry | lblnoise | patchtst | label_noise=0.20 | 4 | 63.19 | 0.62 | 59.11 | -6.62 | — | — |
| industry | lblnoise | residgru | label_noise=0.20 | 4 | 62.39 | 0.32 | 58.08 | -7.42 | — | — |
| industry | lblnoise | cosine | label_noise=0.30 | 4 | 60.81 | 0.28 | 56.27 | -9.00 | — | — |
| industry | lblnoise | patchtst | label_noise=0.30 | 4 | 60.29 | 0.34 | 56.03 | -9.52 | — | — |
| industry | lblnoise | residgru | label_noise=0.30 | 4 | 58.98 | 0.93 | 55.84 | -10.83 | — | — |
| mlspace | O1_single_ckpt | patchtst_single | O1 | 5 | 69.82 | 0.29 | 68.18 | -0.38 | — | — |
| mlspace | O2_blend075_rep | patchtst_top5 | O2 | 6 | 69.99 | 0.44 | 68.34 | -0.21 | — | — |
| mlspace | O3_zero_mean_ema | patchtst_top5_zm09 | O3 | 3 | 69.72 | 0.31 | 68.04 | -0.48 | — | — |
| mlspace | O4_sf_sgd | sf_sgd | sf_lr=0.1 | 3 | 39.54 | 1.42 | 37.80 | -30.27 | — | — |
| mlspace | O4_sf_sgd | sf_sgd | sf_lr=0.3 | 3 | 49.02 | 1.13 | 49.71 | -20.79 | — | — |
| mlspace | O4_sf_sgd | sf_sgd | sf_lr=0.7 | 3 | 57.79 | 0.38 | 56.89 | -12.02 | — | — |
| mlspace | O4_sf_sgd | sf_sgd | sf_lr=1.0 | 3 | 58.93 | 1.39 | 57.78 | -10.88 | — | — |
| mlspace | O5_noise_onset_e15 | cosine | label=0.20@e15 | 4 | 64.50 | 0.10 | 60.49 | -5.31 | — | — |
| mlspace | O5_noise_onset_e15 | patchtst | label=0.20@e15 | 4 | 64.25 | 0.38 | 59.40 | -5.56 | — | — |
| mlspace | O6_long90 | cosine | long90 | 5 | 70.10 | 0.40 | 69.42 | -0.10 | — | — |
| mlspace | O6_long90 | patchtst | long90 | 5 | 69.99 | 0.10 | 69.28 | -0.21 | — | — |
| stars | noshock | cosine | noshock | 4 | 69.90 | 0.11 | 65.75 | +0.09 | — | — |
| stars | noshock | patchtst | noshock | 4 | 69.88 | 0.09 | 65.74 | +0.07 | — | — |
| stars | shock | cosine | shock_lr=0.10 | 4 | 69.45 | 0.52 | 68.33 | -0.36 | — | — |
| stars | shock | patchtst | shock_lr=0.10 | 4 | 70.17 | 0.38 | 65.53 | +0.36 | — | — |
| stars | shock | residgru | shock_lr=0.10 | 4 | 69.31 | 0.49 | 65.50 | -0.50 | — | — |
| stars | shock | cosine | shock_lr=0.50 | 4 | 70.02 | 0.23 | 65.03 | +0.21 | — | — |
| stars | shock | patchtst | shock_lr=0.50 | 8 | 69.53 | 0.35 | 65.36 | -0.28 | — | — |
| stars | shock | residgru | shock_lr=0.50 | 4 | 69.27 | 0.53 | 62.37 | -0.54 | — | — |
| stars | shock | cosine | shock_lr=1.00 | 4 | 69.90 | 0.36 | 61.67 | +0.09 | — | — |
| stars | shock | patchtst | shock_lr=1.00 | 4 | 69.69 | 0.27 | 65.46 | -0.12 | — | — |
| stars | shock | residgru | shock_lr=1.00 | 4 | 62.83 | 7.54 | 56.31 | -6.98 | — | — |
| stars | shock | cosine | shock_lr=2.00 | 4 | 62.14 | 1.30 | 45.88 | -7.67 | — | — |
| stars | shock | patchtst | shock_lr=2.00 | 4 | 59.77 | 1.55 | 52.51 | -10.04 | — | — |
| stars | shock | residgru | shock_lr=2.00 | 4 | 60.19 | 5.38 | 53.90 | -9.62 | — | — |

## 2. Paired Δ(method − cosine) per perturbation cell

Positive Δ means the learnable policy *beats* cosine at that perturbation level.

| experiment | perturb | method | cosine | method | Δ (pp) | cosine_dip | method_dip | dip_red | cos_recov | method_recov |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| O5_noise_onset_e15 | label=0.20@e15 | patchtst | 64.50 | 64.25 | -0.25 | — | — | — | — | — |
| O6_long90 | long90 | patchtst | 70.10 | 69.99 | -0.11 | — | — | — | — | — |
| compound | label=0.20+shock_lr=1.0@e15 | patchtst | 64.08 | 63.11 | -0.98 | — | — | — | — | — |
| compound | label=0.20+shock_lr=1.0@e15 | residgru | 64.08 | 62.30 | -1.78 | — | — | — | — | — |
| innoise | input_noise=0.05 | patchtst | 65.98 | 65.90 | -0.08 | — | — | — | — | — |
| innoise | input_noise=0.15 | patchtst | 57.33 | 56.99 | -0.34 | — | — | — | — | — |
| lblnoise | label_noise=0.10 | patchtst | 66.60 | 66.34 | -0.26 | — | — | — | — | — |
| lblnoise | label_noise=0.10 | residgru | 66.60 | 65.45 | -1.14 | — | — | — | — | — |
| lblnoise | label_noise=0.20 | patchtst | 63.71 | 63.19 | -0.52 | — | — | — | — | — |
| lblnoise | label_noise=0.20 | residgru | 63.71 | 62.39 | -1.32 | — | — | — | — | — |
| lblnoise | label_noise=0.30 | patchtst | 60.81 | 60.29 | -0.52 | — | — | — | — | — |
| lblnoise | label_noise=0.30 | residgru | 60.81 | 58.98 | -1.83 | — | — | — | — | — |
| noshock | noshock | patchtst | 69.90 | 69.88 | -0.02 | — | — | — | — | — |
| shock | shock_lr=0.10 | patchtst | 69.45 | 70.17 | +0.72 | — | — | — | — | — |
| shock | shock_lr=0.10 | residgru | 69.45 | 69.31 | -0.14 | — | — | — | — | — |
| shock | shock_lr=0.50 | patchtst | 70.02 | 69.53 | -0.49 | — | — | — | — | — |
| shock | shock_lr=0.50 | residgru | 70.02 | 69.27 | -0.75 | — | — | — | — | — |
| shock | shock_lr=1.00 | patchtst | 69.90 | 69.69 | -0.21 | — | — | — | — | — |
| shock | shock_lr=1.00 | residgru | 69.90 | 62.83 | -7.07 | — | — | — | — | — |
| shock | shock_lr=2.00 | patchtst | 62.14 | 59.77 | -2.37 | — | — | — | — | — |
| shock | shock_lr=2.00 | residgru | 62.14 | 60.19 | -1.95 | — | — | — | — | — |

## 3. Win/Tie/Loss tally (Δ threshold ±0.30 pp = within seed noise)

- **WIN  (method beats cosine, Δ > +0.30 pp)**: -2
- **TIE  (|Δ| ≤ 0.30 pp)**: 7
- **LOSS (method loses to cosine, Δ < −0.30 pp)**: 13
- Total paired cells: 21

## 4. Open-item resolutions (mlspace)

### Single-checkpoint PatchTST (post-mortem hypothesis)
- `patchtst_single` `O1` (n=5): **69.82 ± 0.29** (Δ vs cos 60ep ref: -0.38 pp)

### PatchTST top-5 blend=0.75 replication (seed-noise bound)
- `patchtst_top5` `O2` (n=6): **69.99 ± 0.44** (Δ vs cos 60ep ref: -0.21 pp)

### Zero-mean EMA ensemble (post-mortem direct fix)
- `patchtst_top5_zm09` `O3` (n=3): **69.72 ± 0.31** (Δ vs cos 60ep ref: -0.48 pp)

### Schedule-Free SGD LR sweep (mis-tuning check)
- `sf_sgd` `sf_lr=0.1` (n=3): **39.54 ± 1.42** (Δ vs cos 35ep ref: -30.27 pp)
- `sf_sgd` `sf_lr=0.3` (n=3): **49.02 ± 1.13** (Δ vs cos 35ep ref: -20.79 pp)
- `sf_sgd` `sf_lr=0.7` (n=3): **57.79 ± 0.38** (Δ vs cos 35ep ref: -12.02 pp)
- `sf_sgd` `sf_lr=1.0` (n=3): **58.93 ± 1.39** (Δ vs cos 35ep ref: -10.88 pp)

## 5. Per-run breakdown

| server | exp | method | perturb | seed | status | final | best@ep | last10 | dip | recov |
|---|---|---|---|---:|---|---:|---|---:|---:|---:|
| industry | compound | cosine | label=0.20+shock_lr=1.0@e15 | 1 | OK | 63.61 | 63.61@35 | 57.27 | — | — |
| industry | compound | cosine | label=0.20+shock_lr=1.0@e15 | 2 | OK | 64.36 | 64.36@35 | 58.01 | — | — |
| industry | compound | cosine | label=0.20+shock_lr=1.0@e15 | 3 | OK | 64.48 | 64.48@35 | 57.76 | — | — |
| industry | compound | cosine | label=0.20+shock_lr=1.0@e15 | 4 | OK | 63.89 | 63.89@35 | 58.32 | — | — |
| industry | compound | patchtst | label=0.20+shock_lr=1.0@e15 | 1 | OK | 62.54 | 62.54@35 | 58.06 | — | — |
| industry | compound | patchtst | label=0.20+shock_lr=1.0@e15 | 2 | OK | 63.63 | 63.63@35 | 59.30 | — | — |
| industry | compound | patchtst | label=0.20+shock_lr=1.0@e15 | 3 | OK | 63.42 | 63.42@35 | 59.56 | — | — |
| industry | compound | patchtst | label=0.20+shock_lr=1.0@e15 | 4 | OK | 62.83 | 62.83@35 | 58.39 | — | — |
| industry | compound | residgru | label=0.20+shock_lr=1.0@e15 | 1 | OK | 62.37 | 62.37@35 | 60.09 | — | — |
| industry | compound | residgru | label=0.20+shock_lr=1.0@e15 | 2 | OK | 62.69 | 62.69@35 | 60.40 | — | — |
| industry | compound | residgru | label=0.20+shock_lr=1.0@e15 | 3 | OK | 61.85 | 61.85@35 | 59.80 | — | — |
| industry | innoise | cosine | input_noise=0.05 | 1 | OK | 65.32 | 65.32@35 | 61.08 | — | — |
| industry | innoise | cosine | input_noise=0.05 | 2 | OK | 65.87 | 65.87@35 | 62.33 | — | — |
| industry | innoise | cosine | input_noise=0.05 | 3 | OK | 66.32 | 66.50@32 | 62.77 | — | — |
| industry | innoise | cosine | input_noise=0.05 | 4 | OK | 66.41 | 66.41@35 | 62.63 | — | — |
| industry | innoise | patchtst | input_noise=0.05 | 1 | OK | 66.48 | 66.48@35 | 63.17 | — | — |
| industry | innoise | patchtst | input_noise=0.05 | 2 | OK | 66.13 | 66.13@35 | 61.67 | — | — |
| industry | innoise | patchtst | input_noise=0.05 | 3 | OK | 65.34 | 65.34@35 | 60.67 | — | — |
| industry | innoise | patchtst | input_noise=0.05 | 4 | OK | 65.65 | 65.66@34 | 62.25 | — | — |
| industry | innoise | cosine | input_noise=0.15 | 1 | OK | 57.19 | 57.19@35 | 52.66 | — | — |
| industry | innoise | cosine | input_noise=0.15 | 2 | OK | 57.74 | 57.74@35 | 54.32 | — | — |
| industry | innoise | cosine | input_noise=0.15 | 3 | OK | 56.80 | 56.80@35 | 52.53 | — | — |
| industry | innoise | cosine | input_noise=0.15 | 4 | OK | 57.59 | 57.59@35 | 55.85 | — | — |
| industry | innoise | patchtst | input_noise=0.15 | 1 | OK | 56.84 | 56.84@35 | 52.59 | — | — |
| industry | innoise | patchtst | input_noise=0.15 | 2 | OK | 57.13 | 57.13@35 | 53.55 | — | — |
| industry | innoise | patchtst | input_noise=0.15 | 3 | OK | 56.94 | 56.94@35 | 53.78 | — | — |
| industry | innoise | patchtst | input_noise=0.15 | 4 | OK | 57.04 | 57.13@33 | 53.88 | — | — |
| industry | lblnoise | cosine | label_noise=0.10 | 1 | OK | 66.19 | 66.19@35 | 62.52 | — | — |
| industry | lblnoise | cosine | label_noise=0.10 | 2 | OK | 66.04 | 66.04@35 | 61.77 | — | — |
| industry | lblnoise | cosine | label_noise=0.10 | 3 | OK | 66.96 | 66.96@35 | 62.50 | — | — |
| industry | lblnoise | cosine | label_noise=0.10 | 4 | OK | 67.20 | 67.20@35 | 63.15 | — | — |
| industry | lblnoise | patchtst | label_noise=0.10 | 1 | OK | 66.47 | 66.47@35 | 61.62 | — | — |
| industry | lblnoise | patchtst | label_noise=0.10 | 2 | OK | 66.31 | 66.31@35 | 61.77 | — | — |
| industry | lblnoise | patchtst | label_noise=0.10 | 3 | OK | 66.59 | 66.59@35 | 62.44 | — | — |
| industry | lblnoise | patchtst | label_noise=0.10 | 4 | OK | 65.99 | 65.99@35 | 62.24 | — | — |
| industry | lblnoise | residgru | label_noise=0.10 | 1 | OK | 64.91 | 64.91@35 | 62.89 | — | — |
| industry | lblnoise | residgru | label_noise=0.10 | 2 | OK | 65.39 | 65.39@35 | 60.08 | — | — |
| industry | lblnoise | residgru | label_noise=0.10 | 3 | OK | 66.47 | 66.47@35 | 59.76 | — | — |
| industry | lblnoise | residgru | label_noise=0.10 | 4 | OK | 65.04 | 65.04@35 | 62.69 | — | — |
| industry | lblnoise | cosine | label_noise=0.20 | 1 | OK | 64.05 | 64.05@35 | 58.48 | — | — |
| industry | lblnoise | cosine | label_noise=0.20 | 2 | OK | 63.61 | 63.61@35 | 58.63 | — | — |
| industry | lblnoise | cosine | label_noise=0.20 | 3 | OK | 63.75 | 63.75@35 | 59.12 | — | — |
| industry | lblnoise | cosine | label_noise=0.20 | 4 | OK | 63.43 | 63.43@35 | 59.52 | — | — |
| industry | lblnoise | patchtst | label_noise=0.20 | 1 | OK | 62.21 | 62.27@34 | 58.76 | — | — |
| industry | lblnoise | patchtst | label_noise=0.20 | 2 | OK | 63.93 | 63.93@35 | 59.49 | — | — |
| industry | lblnoise | patchtst | label_noise=0.20 | 3 | OK | 63.33 | 63.33@35 | 59.01 | — | — |
| industry | lblnoise | patchtst | label_noise=0.20 | 4 | OK | 63.27 | 63.27@35 | 59.19 | — | — |
| industry | lblnoise | residgru | label_noise=0.20 | 1 | OK | 62.68 | 62.68@35 | 59.86 | — | — |
| industry | lblnoise | residgru | label_noise=0.20 | 2 | OK | 62.50 | 62.50@35 | 59.06 | — | — |
| industry | lblnoise | residgru | label_noise=0.20 | 3 | OK | 62.53 | 62.53@35 | 54.39 | — | — |
| industry | lblnoise | residgru | label_noise=0.20 | 4 | OK | 61.85 | 61.85@35 | 59.01 | — | — |
| industry | lblnoise | cosine | label_noise=0.30 | 1 | OK | 60.38 | 60.38@35 | 56.04 | — | — |
| industry | lblnoise | cosine | label_noise=0.30 | 2 | OK | 60.73 | 60.73@35 | 55.57 | — | — |
| industry | lblnoise | cosine | label_noise=0.30 | 3 | OK | 61.02 | 61.02@35 | 56.88 | — | — |
| industry | lblnoise | cosine | label_noise=0.30 | 4 | OK | 61.10 | 61.10@35 | 56.58 | — | — |
| industry | lblnoise | patchtst | label_noise=0.30 | 1 | OK | 60.37 | 60.37@35 | 55.95 | — | — |
| industry | lblnoise | patchtst | label_noise=0.30 | 2 | OK | 59.80 | 59.80@35 | 55.38 | — | — |
| industry | lblnoise | patchtst | label_noise=0.30 | 3 | OK | 60.25 | 60.25@35 | 55.90 | — | — |
| industry | lblnoise | patchtst | label_noise=0.30 | 4 | OK | 60.74 | 60.74@35 | 56.88 | — | — |
| industry | lblnoise | residgru | label_noise=0.30 | 1 | OK | 57.89 | 58.16@34 | 56.09 | — | — |
| industry | lblnoise | residgru | label_noise=0.30 | 2 | OK | 59.01 | 59.01@35 | 56.64 | — | — |
| industry | lblnoise | residgru | label_noise=0.30 | 3 | OK | 60.43 | 60.43@35 | 53.82 | — | — |
| industry | lblnoise | residgru | label_noise=0.30 | 4 | OK | 58.58 | 58.63@34 | 56.81 | — | — |
| mlspace | O1_single_ckpt | patchtst_single | O1 | 1 | OK | 70.13 | 70.42@59 | 68.72 | — | — |
| mlspace | O1_single_ckpt | patchtst_single | O1 | 2 | OK | 69.77 | 69.89@59 | 67.84 | — | — |
| mlspace | O1_single_ckpt | patchtst_single | O1 | 3 | OK | 69.60 | 69.60@60 | 67.71 | — | — |
| mlspace | O1_single_ckpt | patchtst_single | O1 | 4 | OK | 70.16 | 70.18@58 | 68.68 | — | — |
| mlspace | O1_single_ckpt | patchtst_single | O1 | 5 | OK | 69.43 | 69.50@59 | 67.94 | — | — |
| mlspace | O2_blend075_rep | patchtst_top5 | O2 | 4 | OK | 70.54 | 70.54@60 | 68.70 | — | — |
| mlspace | O2_blend075_rep | patchtst_top5 | O2 | 5 | OK | 69.64 | 69.64@60 | 68.12 | — | — |
| mlspace | O2_blend075_rep | patchtst_top5 | O2 | 6 | OK | 70.45 | 70.45@60 | 68.89 | — | — |
| mlspace | O2_blend075_rep | patchtst_top5 | O2 | 7 | OK | 69.57 | 69.87@59 | 67.91 | — | — |
| mlspace | O2_blend075_rep | patchtst_top5 | O2 | 8 | OK | 69.47 | 69.47@60 | 67.72 | — | — |
| mlspace | O2_blend075_rep | patchtst_top5 | O2 | 9 | OK | 70.27 | 70.27@60 | 68.68 | — | — |
| mlspace | O3_zero_mean_ema | patchtst_top5_zm09 | O3 | 1 | OK | 69.90 | 69.90@60 | 68.39 | — | — |
| mlspace | O3_zero_mean_ema | patchtst_top5_zm09 | O3 | 2 | OK | 69.28 | 69.28@60 | 67.56 | — | — |
| mlspace | O3_zero_mean_ema | patchtst_top5_zm09 | O3 | 3 | OK | 69.97 | 69.97@60 | 68.16 | — | — |
| mlspace | O4_sf_sgd | sf_sgd | sf_lr=0.1 | 1 | OK | 37.67 | 38.66@34 | 37.05 | — | — |
| mlspace | O4_sf_sgd | sf_sgd | sf_lr=0.1 | 2 | OK | 39.84 | 39.84@35 | 37.92 | — | — |
| mlspace | O4_sf_sgd | sf_sgd | sf_lr=0.1 | 3 | OK | 41.12 | 41.12@35 | 38.44 | — | — |
| mlspace | O4_sf_sgd | sf_sgd | sf_lr=0.3 | 1 | OK | 49.44 | 51.05@27 | 50.49 | — | — |
| mlspace | O4_sf_sgd | sf_sgd | sf_lr=0.3 | 2 | OK | 47.48 | 50.17@24 | 48.55 | — | — |
| mlspace | O4_sf_sgd | sf_sgd | sf_lr=0.3 | 3 | OK | 50.14 | 51.02@29 | 50.08 | — | — |
| mlspace | O4_sf_sgd | sf_sgd | sf_lr=0.7 | 1 | OK | 58.24 | 58.62@31 | 57.45 | — | — |
| mlspace | O4_sf_sgd | sf_sgd | sf_lr=0.7 | 2 | OK | 57.32 | 58.87@31 | 57.45 | — | — |
| mlspace | O4_sf_sgd | sf_sgd | sf_lr=0.7 | 3 | OK | 57.81 | 57.81@35 | 55.76 | — | — |
| mlspace | O4_sf_sgd | sf_sgd | sf_lr=1.0 | 1 | OK | 58.15 | 58.77@32 | 56.03 | — | — |
| mlspace | O4_sf_sgd | sf_sgd | sf_lr=1.0 | 2 | OK | 60.88 | 60.88@35 | 59.08 | — | — |
| mlspace | O4_sf_sgd | sf_sgd | sf_lr=1.0 | 3 | OK | 57.76 | 60.06@34 | 58.24 | — | — |
| mlspace | O5_noise_onset_e15 | cosine | label=0.20@e15 | 1 | OK | 64.42 | 64.42@35 | 59.99 | — | — |
| mlspace | O5_noise_onset_e15 | cosine | label=0.20@e15 | 2 | OK | 64.63 | 64.63@35 | 60.90 | — | — |
| mlspace | O5_noise_onset_e15 | cosine | label=0.20@e15 | 3 | OK | 64.38 | 64.38@35 | 60.38 | — | — |
| mlspace | O5_noise_onset_e15 | cosine | label=0.20@e15 | 4 | OK | 64.58 | 64.65@34 | 60.69 | — | — |
| mlspace | O5_noise_onset_e15 | patchtst | label=0.20@e15 | 1 | OK | 64.40 | 64.40@35 | 57.14 | — | — |
| mlspace | O5_noise_onset_e15 | patchtst | label=0.20@e15 | 2 | OK | 63.66 | 63.79@34 | 59.93 | — | — |
| mlspace | O5_noise_onset_e15 | patchtst | label=0.20@e15 | 3 | OK | 64.71 | 64.71@35 | 60.60 | — | — |
| mlspace | O5_noise_onset_e15 | patchtst | label=0.20@e15 | 4 | OK | 64.25 | 64.25@35 | 59.92 | — | — |
| mlspace | O6_long90 | cosine | long90 | 1 | OK | 69.46 | 69.46@88 | 69.02 | — | — |
| mlspace | O6_long90 | cosine | long90 | 2 | OK | 70.30 | 70.43@89 | 69.50 | — | — |
| mlspace | O6_long90 | cosine | long90 | 3 | OK | 69.81 | 69.96@89 | 69.30 | — | — |
| mlspace | O6_long90 | cosine | long90 | 4 | OK | 70.44 | 70.55@89 | 69.58 | — | — |
| mlspace | O6_long90 | cosine | long90 | 5 | OK | 70.49 | 70.63@89 | 69.71 | — | — |
| mlspace | O6_long90 | patchtst | long90 | 1 | OK | 70.07 | 70.08@89 | 69.39 | — | — |
| mlspace | O6_long90 | patchtst | long90 | 2 | OK | 69.98 | 69.98@90 | 69.46 | — | — |
| mlspace | O6_long90 | patchtst | long90 | 3 | OK | 70.01 | 70.08@88 | 69.08 | — | — |
| mlspace | O6_long90 | patchtst | long90 | 4 | OK | 70.07 | 70.12@89 | 69.48 | — | — |
| mlspace | O6_long90 | patchtst | long90 | 5 | OK | 69.81 | 69.86@88 | 68.98 | — | — |
| stars | noshock | cosine | noshock | 5 | OK | 69.85 | 69.85@35 | 66.18 | — | — |
| stars | noshock | cosine | noshock | 6 | OK | 69.84 | 69.84@35 | 65.73 | — | — |
| stars | noshock | cosine | noshock | 7 | OK | 70.08 | 70.08@35 | 65.63 | — | — |
| stars | noshock | cosine | noshock | 8 | OK | 69.82 | 69.82@35 | 65.47 | — | — |
| stars | noshock | patchtst | noshock | 5 | OK | 69.75 | 69.75@35 | 66.19 | — | — |
| stars | noshock | patchtst | noshock | 6 | OK | 69.99 | 69.99@35 | 64.94 | — | — |
| stars | noshock | patchtst | noshock | 7 | OK | 69.91 | 69.91@35 | 65.83 | — | — |
| stars | noshock | patchtst | noshock | 8 | OK | 69.88 | 69.88@35 | 66.00 | — | — |
| stars | shock | cosine | shock_lr=0.10 | 1 | OK | 69.94 | 69.94@35 | 69.01 | — | — |
| stars | shock | cosine | shock_lr=0.10 | 2 | OK | 69.58 | 69.58@35 | 68.53 | — | — |
| stars | shock | cosine | shock_lr=0.10 | 3 | OK | 69.71 | 69.71@35 | 68.28 | — | — |
| stars | shock | cosine | shock_lr=0.10 | 4 | OK | 68.57 | 68.57@35 | 67.48 | — | — |
| stars | shock | patchtst | shock_lr=0.10 | 1 | OK | 69.51 | 69.51@35 | 64.64 | — | — |
| stars | shock | patchtst | shock_lr=0.10 | 2 | OK | 70.30 | 70.30@35 | 66.33 | — | — |
| stars | shock | patchtst | shock_lr=0.10 | 3 | OK | 70.44 | 70.44@35 | 65.32 | — | — |
| stars | shock | patchtst | shock_lr=0.10 | 4 | OK | 70.42 | 70.42@35 | 65.83 | — | — |
| stars | shock | residgru | shock_lr=0.10 | 1 | OK | 69.70 | 69.70@35 | 62.22 | — | — |
| stars | shock | residgru | shock_lr=0.10 | 2 | OK | 69.87 | 69.87@35 | 66.40 | — | — |
| stars | shock | residgru | shock_lr=0.10 | 3 | OK | 68.72 | 68.73@34 | 66.77 | — | — |
| stars | shock | residgru | shock_lr=0.10 | 4 | OK | 68.95 | 68.95@35 | 66.61 | — | — |
| stars | shock | cosine | shock_lr=0.50 | 1 | OK | 69.69 | 69.69@35 | 64.51 | — | — |
| stars | shock | cosine | shock_lr=0.50 | 2 | OK | 69.93 | 69.93@35 | 65.40 | — | — |
| stars | shock | cosine | shock_lr=0.50 | 3 | OK | 70.14 | 70.19@34 | 64.84 | — | — |
| stars | shock | cosine | shock_lr=0.50 | 4 | OK | 70.31 | 70.31@35 | 65.36 | — | — |
| stars | shock | patchtst | shock_lr=0.50 | 1 | OK | 69.84 | 69.84@35 | 65.82 | — | — |
| stars | shock | patchtst | shock_lr=0.50 | 2 | OK | 69.56 | 69.56@35 | 64.90 | — | — |
| stars | shock | patchtst | shock_lr=0.50 | 3 | OK | 68.96 | 68.96@35 | 65.12 | — | — |
| stars | shock | patchtst | shock_lr=0.50 | 4 | OK | 69.51 | 69.51@35 | 65.60 | — | — |
| stars | shock | patchtst | shock_lr=0.50 | 5 | OK | 69.68 | 69.68@35 | 65.80 | — | — |
| stars | shock | patchtst | shock_lr=0.50 | 6 | OK | 69.98 | 69.98@35 | 65.37 | — | — |
| stars | shock | patchtst | shock_lr=0.50 | 7 | OK | 69.71 | 69.71@35 | 65.35 | — | — |
| stars | shock | patchtst | shock_lr=0.50 | 8 | OK | 68.99 | 69.20@34 | 64.91 | — | — |
| stars | shock | residgru | shock_lr=0.50 | 1 | OK | 69.77 | 69.77@35 | 62.67 | — | — |
| stars | shock | residgru | shock_lr=0.50 | 2 | OK | 68.44 | 68.44@35 | 62.05 | — | — |
| stars | shock | residgru | shock_lr=0.50 | 3 | OK | 69.67 | 69.67@35 | 63.46 | — | — |
| stars | shock | residgru | shock_lr=0.50 | 4 | OK | 69.20 | 69.20@35 | 61.27 | — | — |
| stars | shock | cosine | shock_lr=1.00 | 1 | OK | 69.40 | 69.40@35 | 61.34 | — | — |
| stars | shock | cosine | shock_lr=1.00 | 2 | OK | 69.78 | 69.78@35 | 61.99 | — | — |
| stars | shock | cosine | shock_lr=1.00 | 3 | OK | 70.04 | 70.04@35 | 62.50 | — | — |
| stars | shock | cosine | shock_lr=1.00 | 4 | OK | 70.37 | 70.37@35 | 60.85 | — | — |
| stars | shock | patchtst | shock_lr=1.00 | 1 | OK | 69.99 | 69.99@35 | 66.13 | — | — |
| stars | shock | patchtst | shock_lr=1.00 | 2 | OK | 69.71 | 69.71@35 | 65.08 | — | — |
| stars | shock | patchtst | shock_lr=1.00 | 3 | OK | 69.25 | 69.25@35 | 64.87 | — | — |
| stars | shock | patchtst | shock_lr=1.00 | 4 | OK | 69.80 | 69.80@35 | 65.75 | — | — |
| stars | shock | residgru | shock_lr=1.00 | 1 | OK | 51.39 | 59.57@19 | 41.41 | — | — |
| stars | shock | residgru | shock_lr=1.00 | 2 | OK | 69.54 | 69.54@35 | 66.68 | — | — |
| stars | shock | residgru | shock_lr=1.00 | 3 | OK | 69.67 | 69.67@35 | 60.45 | — | — |
| stars | shock | residgru | shock_lr=1.00 | 4 | OK | 60.71 | 60.71@35 | 56.73 | — | — |
| stars | shock | cosine | shock_lr=2.00 | 1 | OK | 60.22 | 60.22@35 | 46.62 | — | — |
| stars | shock | cosine | shock_lr=2.00 | 2 | OK | 62.62 | 62.62@35 | 45.03 | — | — |
| stars | shock | cosine | shock_lr=2.00 | 3 | OK | 61.92 | 61.92@35 | 42.72 | — | — |
| stars | shock | cosine | shock_lr=2.00 | 4 | OK | 63.80 | 63.80@35 | 49.13 | — | — |
| stars | shock | patchtst | shock_lr=2.00 | 1 | OK | 60.34 | 60.34@35 | 52.32 | — | — |
| stars | shock | patchtst | shock_lr=2.00 | 2 | OK | 60.14 | 60.14@35 | 52.24 | — | — |
| stars | shock | patchtst | shock_lr=2.00 | 3 | OK | 57.22 | 57.22@35 | 49.57 | — | — |
| stars | shock | patchtst | shock_lr=2.00 | 4 | OK | 61.39 | 61.39@35 | 55.91 | — | — |
| stars | shock | residgru | shock_lr=2.00 | 1 | OK | 68.27 | 68.27@35 | 59.69 | — | — |
| stars | shock | residgru | shock_lr=2.00 | 2 | OK | 57.43 | 57.43@35 | 51.68 | — | — |
| stars | shock | residgru | shock_lr=2.00 | 3 | OK | 53.75 | 57.93@19 | 49.19 | — | — |
| stars | shock | residgru | shock_lr=2.00 | 4 | OK | 61.33 | 61.33@35 | 55.04 | — | — |
