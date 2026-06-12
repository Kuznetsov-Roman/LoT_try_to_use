# 2-hour Online MPC Curve Policy POC — consolidated results

Generated: 2026-05-12T19:51:18
Seed: 400
Cosine 4-seed reference: 69.81 +/- 0.33

## Variant final accuracy

| variant | final_acc (%) | delta vs cosine_ref (pp) | best_acc (%) @ epoch | final_loss |
|---|---|---|---|---|
| `cosine_ref` | 70.640 | +0.000 | 70.640 @ 35 | 1.0825 |
| `curve_frozen` | 32.730 | -37.910 | 42.030 @ 6 | 2.8160 |
| `curve_online` | 31.730 | -38.910 | 42.100 @ 14 | 3.0639 |

## Verdict

**NULL**: Online curve policy is much worse than cosine on accuracy. However, online supervision did reduce curve MSE substantially (decline 88.8%) and argmin-LR error decreased over the run, suggesting the predictive mechanism works but the argmin signal saturates at a destructive LR.

## Curve prediction quality (curve variants)

| variant | mean_curve_mse | mse_first/second half | mean_rel_lr_err | rel_lr_err first/second half |
|---|---|---|---|---|
| `curve_online` | 0.3814 | 0.6621 / 0.1163 | 2.272 | 4.271 / 0.384 |
| `curve_frozen` | 0.4931 | 0.7926 / 0.2103 | 2.061 | 3.839 / 0.381 |

### argmin index trajectory (LR_GRID has 30 entries, max idx 29 -> LR=2.5)

- `curve_online`: pred_argmin_idx first=18.1 last=23.0; real_argmin_idx first=2.7 last=29.0
- `curve_frozen`: pred_argmin_idx first=18.1 last=23.0; real_argmin_idx first=2.9 last=29.0

## Online learning loss decline (curve variants)

| variant | epochs_with_loss | first | last | mean | decline (%) |
|---|---|---|---|---|---|
| `curve_online` | 34 | 1.7490 | 0.1951 | 0.2897 | 88.8 |
| `curve_frozen` | 34 | 1.4551 | 0.1822 | 0.2646 | 87.5 |

## Applied LR statistics

| variant | min | max | mean | first | last |
|---|---|---|---|---|---|
| `cosine_ref` | 0.0000 | 0.9980 | 0.4857 | 0.9980 | 0.0000 |
| `curve_frozen` | 0.9505 | 1.8987 | 1.4399 | 0.9980 | 1.5012 |
| `curve_online` | 0.9505 | 1.8965 | 1.3946 | 0.9980 | 1.4900 |

## Offline curve-policy pretraining (oracle_v2, lookahead_n=2)

- model_type=`curve` input_dim=233 window=10 lookahead_n=2
- best_val_mse=0.004228 test_mse_total=0.015457 test_mse_per_step=[0.014603380839029947, 0.016310476938883465]
- test_argmin_lr_rel_error=0.2697 (train_windows=300 val_windows=50 test_windows=50)

## Plots

- ![accuracy_seed400.png](logs/2h_curve/plots/accuracy_seed400.png)
- ![curve_error_seed400.png](logs/2h_curve/plots/curve_error_seed400.png)
- ![curve_overlay_seed400.png](logs/2h_curve/plots/curve_overlay_seed400.png)
- ![lr_trajectory_seed400.png](logs/2h_curve/plots/lr_trajectory_seed400.png)

## Artifact paths (under this folder)

- `logs/2h_curve/summary.md` — original summarize_2h_curve.py output
- `logs/2h_curve/lr_2h_curve.log` — orchestrator log with PHASE_START/PHASE_DONE markers
- `logs/2h_curve/{cosine_ref,curve_n2_frozen,curve_n2_online}_seed400.log` — per-variant deploy stdout
- `logs/2h_curve/curve_n2_online_seed400.log.failed_*` — pre-fix crash log (autograd bug)
- `snapshots/2h_curve/<variant>/metrics.jsonl` — per-epoch student/teacher metrics
- `snapshots/2h_curve/<variant>/lr_data_epoch_*.pt` — per-epoch applied LR
- `snapshots/2h_curve/<variant>/curve_online.jsonl` — per-epoch curve diagnostics (curve variants only)
- `snapshots/2h_curve/<variant>/predicted_curves_epoch_*.npy` — N x 30 forecast tensors
- `snapshots/2h_curve/<variant>/real_curve_epoch_*.npy` — 30 real probed losses per epoch
- `checkpoints/policies/curve_v2_n2/{policy.pt,summary.json,history.jsonl}` — pretrained checkpoint and offline training history
- `per_epoch.csv` — long-format per-variant per-epoch table
- `curve_online.csv` — long-format per-epoch curve diagnostics
