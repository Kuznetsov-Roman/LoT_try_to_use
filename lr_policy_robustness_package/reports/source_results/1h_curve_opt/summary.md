# Curve Policy run summary

Generated: 2026-05-12T21:10:58

Seed: 400

Cosine 4-seed reference: 69.81 +/- 0.33 (prior runs, depth_list=110_20).

## Final accuracy

- `curve_frozen_opt`: final_acc=50.160 delta_vs_cosine=-19.650 pp final_loss=1.9310
- `curve_full_opt`: final_acc=36.800 delta_vs_cosine=-33.010 pp final_loss=2.7791
- `curve_parab_opt`: final_acc=41.430 delta_vs_cosine=-28.380 pp final_loss=2.3208

## Curve prediction quality (curve variants only)

- `curve_frozen_opt`: epochs=29 mean_curve_mse=0.389827 mean_rel_lr_err=2.7668 mse_first_half=0.779508 mse_second_half=0.026126 rel_lr_err_first=5.2138 rel_lr_err_second=0.4830
- `curve_full_opt`: epochs=29 mean_curve_mse=0.456073 mean_rel_lr_err=2.1944 mse_first_half=0.882366 mse_second_half=0.058199 rel_lr_err_first=4.2845 rel_lr_err_second=0.2435
- `curve_parab_opt`: epochs=29 mean_curve_mse=0.453442 mean_rel_lr_err=2.4704 mse_first_half=0.877202 mse_second_half=0.057932 rel_lr_err_first=4.8645 rel_lr_err_second=0.2359

## Online learning loss trajectory

- `curve_frozen_opt`: epochs_with_loss=28 first_loss=1.612772 last_loss=0.064945 mean_loss=0.142068 relative_decline=96.0%
- `curve_full_opt`: epochs_with_loss=28 first_loss=2.236452 last_loss=0.091879 mean_loss=0.322298 relative_decline=95.9%
- `curve_parab_opt`: epochs_with_loss=28 first_loss=1.947877 last_loss=0.174918 mean_loss=0.313953 relative_decline=91.0%

## Per-variant LR trajectory summary

- `curve_frozen_opt`: lr_min=0.4180 lr_max=0.9980 lr_mean=0.8049 lr_first=0.9980 lr_last=0.4180
- `curve_full_opt`: lr_min=0.5983 lr_max=0.9980 lr_mean=0.7879 lr_first=0.9980 lr_last=0.6359
- `curve_parab_opt`: lr_min=0.5551 lr_max=0.9980 lr_mean=0.7934 lr_first=0.9980 lr_last=0.6359

## Verdict

**NULL**: best curve variant `curve_frozen_opt` is -19.650 pp vs cosine. Likely next: argmin-KL loss, curve normalisation, or regenerate oracle in deploy regime.

## Plots

- results/1h_curve_opt/logs/1h_curve_opt\plots\accuracy_seed400.png
- results/1h_curve_opt/logs/1h_curve_opt\plots\lr_trajectory_seed400.png
- results/1h_curve_opt/logs/1h_curve_opt\plots\curve_error_seed400.png
