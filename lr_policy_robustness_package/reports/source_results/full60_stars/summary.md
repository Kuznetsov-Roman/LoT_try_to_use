# 60-Epoch Full Run — cosine vs P3 residual_log

- Server: stars (A100-80GB), tmux `full60`
- Bundles: 3-parallel residual_log (~2h 23min) + 3-parallel cosine (~2h 22min)
- Final epoch: **60**, seeds: 1, 2, 3

## Final-epoch student/teacher metrics (mean +/- std over 3 seeds)

| Method | Student Test Acc | Student Test Loss | Teacher Test Acc | Teacher Test Loss |
|---|---|---|---|---|
| cosine | 70.20 +/- 0.23 | 1.121 +/- 0.007 | 76.28 +/- 0.23 | 0.833 +/- 0.004 |
| residual_log | 69.81 +/- 0.21 | 1.134 +/- 0.010 | 76.30 +/- 0.48 | 0.836 +/- 0.009 |

## Per-seed best student test accuracy (over all epochs)

| Method | Seed | Best Acc | @epoch |
|---|---|---|---|
| cosine | 1 | 70.00 | 59 |
| cosine | 2 | 70.15 | 60 |
| cosine | 3 | 70.45 | 60 |
| residual_log | 1 | 69.70 | 59 |
| residual_log | 2 | 70.00 | 59 |
| residual_log | 3 | 69.98 | 60 |

## Method-aggregated best student test acc (mean +/- std)

| Method | Mean Best Acc | Std | n_seeds |
|---|---|---|---|
| cosine | 70.20 | 0.23 | 3 |
| residual_log | 69.89 | 0.17 | 3 |

## Last-10-epoch averaged student test acc (smoothed final)

| Method | Mean Acc (last 10 ep) | Std | n_seeds |
|---|---|---|---|
| cosine | 68.42 | 0.35 | 3 |
| residual_log | 68.47 | 0.57 | 3 |
