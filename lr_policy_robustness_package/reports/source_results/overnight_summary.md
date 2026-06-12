# Overnight LR Policy Summary

Generated: 2026-05-11T03:19:52

All jobs were launched from a single remote tmux orchestrator session.

## Oracle Dataset v2

- Rows: 576
- Feature dimension: 230
- Trajectories: 16

## Baseline Cosine Seeds

- `overnight_baseline_seed1`: epoch=35 student_acc=70.110 student_loss=1.0809 teacher_acc=75.160
- `overnight_baseline_seed2`: epoch=35 student_acc=69.980 student_loss=1.0852 teacher_acc=75.030
- `overnight_baseline_seed3`: epoch=35 student_acc=69.350 student_loss=1.1007 teacher_acc=75.060

## Current GRU Policy Eval

- `overnight_current_gru_seed31`: epoch=35 student_acc=34.450 student_loss=2.7333 teacher_acc=74.450
- `overnight_current_gru_seed32`: epoch=35 student_acc=42.780 student_loss=2.2706 teacher_acc=74.410
- `overnight_current_gru_seed33`: epoch=35 student_acc=44.340 student_loss=2.2289 teacher_acc=74.300

## Modular Policy Eval

- `overnight_modular_seed41`: epoch=35 student_acc=69.050 student_loss=1.1129 teacher_acc=74.920
- `overnight_modular_seed42`: epoch=35 student_acc=69.010 student_loss=1.1239 teacher_acc=74.780
- `overnight_modular_seed43`: epoch=35 student_acc=68.290 student_loss=1.1295 teacher_acc=74.900

## Offline Policy Fits

- `current_gru_targets_multiplier`: target=targets_multiplier test_mse=0.143847 test_mae=0.281071
- `current_gru_targets_raw_lr`: target=targets_raw_lr test_mse=0.050814 test_mae=0.178463
- `current_gru_targets_smoothed_multiplier`: target=targets_smoothed_multiplier test_mse=0.066050 test_mae=0.205259
- `modular_targets_smoothed_multiplier`: target=targets_smoothed_multiplier test_mse=0.024785 test_mae=0.116084

## Recommendation

Prefer the model/target combination with the best online student accuracy among completed runs. If online metrics are close, prefer the smoothed multiplier target because it is less sensitive to one-step oracle noise.
