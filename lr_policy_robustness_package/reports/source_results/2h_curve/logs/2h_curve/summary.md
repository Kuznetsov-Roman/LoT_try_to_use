# 2-hour Online MPC Curve Policy POC

Generated: 2026-05-12T13:12:06

Seed: 400

Cosine 4-seed reference: 69.81 +/- 0.33 (prior runs, depth_list=110_20).

## Final accuracy

- `cosine_ref`: final_acc=70.640 delta_vs_cosine=+0.830 pp final_loss=1.0825
- `curve_frozen`: final_acc=32.730 delta_vs_cosine=-37.080 pp final_loss=2.8160
- `curve_online`: final_acc=31.730 delta_vs_cosine=-38.080 pp final_loss=3.0639

## Curve prediction quality (curve variants only)

- `curve_online`: epochs=35 mean_curve_mse=0.381429 mean_rel_lr_err=2.2721 mse_first_half=0.688249 mse_second_half=0.091655 rel_lr_err_first=4.2558 rel_lr_err_second=0.3987
- `curve_frozen`: epochs=35 mean_curve_mse=0.493127 mean_rel_lr_err=2.0605 mse_first_half=0.836415 mse_second_half=0.168911 rel_lr_err_first=3.8166 rel_lr_err_second=0.4020

## Online learning loss trajectory

- `curve_online`: epochs_with_loss=34 first_loss=1.748957 last_loss=0.195130 mean_loss=0.289732 relative_decline=88.8%
- `curve_frozen`: epochs_with_loss=34 first_loss=1.455128 last_loss=0.182240 mean_loss=0.264582 relative_decline=87.5%

## Per-variant LR trajectory summary

- `curve_online`: lr_min=0.9505 lr_max=1.8965 lr_mean=1.3946 lr_first=0.9980 lr_last=1.4900
- `curve_frozen`: lr_min=0.9505 lr_max=1.8987 lr_mean=1.4399 lr_first=0.9980 lr_last=1.5012
- `cosine_ref`: lr_min=0.0000 lr_max=0.9980 lr_mean=0.4857 lr_first=0.9980 lr_last=0.0000

## Verdict

**NULL**: curve_online does not match cosine within 1.0 pp. Inspect predicted-vs-real overlay PNGs, online learning curve, and whether argmin saturates. Likely next: stronger pretraining, curve normalisation (z-score per epoch), or auxiliary argmin-KL loss.

## Plots

- logs/2h_curve/plots/accuracy_seed400.png
- logs/2h_curve/plots/lr_trajectory_seed400.png
- logs/2h_curve/plots/curve_error_seed400.png
- logs/2h_curve/plots/curve_overlay_seed400.png
