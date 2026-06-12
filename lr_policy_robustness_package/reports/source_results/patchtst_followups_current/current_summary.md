# PatchTST Follow-ups — Current Results

Generated from pulled logs. Runs that are still active are marked by `last_epoch < target`.

## Per-run latest/best metrics

| phase   | method   |   seed |   last_epoch |   last_acc |   last_loss |   best_acc |   best_epoch |   last_student_lr |
|:--------|:---------|-------:|-------------:|-----------:|------------:|-----------:|-------------:|------------------:|
| 60ep    | patchtst |      1 |           50 |      63.5  |       1.362 |      63.5  |           50 |            0.1089 |
| 60ep    | patchtst |      2 |           50 |      64.56 |       1.308 |      64.56 |           50 |            0.1091 |
| 60ep    | patchtst |      3 |           50 |      63.05 |       1.364 |      63.05 |           50 |            0.1091 |
| base35  | patchtst |     20 |           35 |      69.83 |       1.087 |      69.83 |           35 |            0.004  |
| base35  | patchtst |     21 |           35 |      69.7  |       1.085 |      69.7  |           35 |            0.004  |
| base35  | patchtst |     22 |           35 |      69.51 |       1.1   |      69.51 |           35 |            0.004  |
| ext50   | patchtst |     20 |           49 |      69.56 |       1.128 |      69.83 |           35 |            0.0066 |
| ext50   | patchtst |     21 |           49 |      69.53 |       1.114 |      69.7  |           35 |            0.0066 |
| ext50   | patchtst |     22 |           49 |      69.12 |       1.147 |      69.59 |           36 |            0.0066 |
| fixed35 | dlinear  |     20 |           35 |      69.56 |       1.124 |      69.56 |           35 |            0.0011 |
| fixed35 | nbeats   |     20 |           35 |      69.64 |       1.129 |      69.64 |           35 |            0.0011 |

## Fixed NBeats/DLinear result

- `dlinear` seed 20: epoch 35, last_acc=69.56, best_acc=69.56@35
- `nbeats` seed 20: epoch 35, last_acc=69.64, best_acc=69.64@35

## PatchTST warm-restart progress

- Current epochs: 49, 49, 49; mean current acc=69.40.

## PatchTST 60ep progress

- Current epochs: 50, 50, 50; mean current acc=63.70.

## Figures

- `results/figures/patchtst_60ep_current.png`
- `results/figures/nbeats_dlinear_fixed35.png`
- `results/figures/patchtst_ext50_current.png`
