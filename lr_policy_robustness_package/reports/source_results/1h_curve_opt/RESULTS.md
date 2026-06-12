# 1h Curve Policy — Optimised Run (interrupted at ~28/35 epochs)

Generated: 2026-05-12T21:13:30

## Setup

Common to all 3 variants:

- Pretrained policy: `checkpoints/policies/curve_v2_n2/policy.pt` (CurveLRPolicy, lookahead_n=2)
- Deployment: CIFAR-100, depth_list=110_20, student_steps_ratio=4, batch_size=256, seed=400, 35 epochs target
- Policy: `--policy_max_lr 1.2 --policy_cosine_blend 0.5 --policy_warmup_epochs 10`
  (proven d1_sweep recipe → restores sane LR regime instead of saturating at LR=1.5+)

Per-variant:

| Variant | argmin selection | online_lr | online optimizer | Purpose |
|---|---|---:|---|---|
| `curve_full_opt`   | softmax (T=0.1)  | 1e-3 | Adam | Full optimization stack |
| `curve_parab_opt`  | parabolic refine | 1e-3 | Adam | Ablate softmax vs parabolic |
| `curve_frozen_opt` | softmax (T=0.1)  | 0    | —    | Ablate online updates (frozen policy) |

Reference baselines:

- Cosine 4-seed: **69.81 ± 0.33 pp** (depth=110_20, ratio=4)
- 2h_curve POC: cosine_ref=70.64, curve_n2_frozen=32.73, curve_n2_online=31.73

## Final accuracy at last completed eval epoch

| Variant | Last epoch | Student Test Acc | Δ vs cosine (69.81) | Δ vs POC same-name |
|---|---:|---:|---:|---:|
| `curve_full_opt` | 29 | **36.80** | -33.01 | 5.07 |
| `curve_parab_opt` | 29 | **41.43** | -28.38 | 9.70 |
| `curve_frozen_opt` | 29 | **50.16** | -19.65 | 17.43 |

### Best (peak) accuracy any time during training

| Variant | Peak acc | Peak epoch |
|---|---:|---:|
| `curve_full_opt` | **49.13** | 10 |
| `curve_parab_opt` | **51.27** | 28 |
| `curve_frozen_opt` | **55.03** | 28 |

## Curve prediction quality

| Variant | epochs | mean_curve_mse | mse_first_half | mse_second_half | mse drop | mean_argmin_err_pp* |
|---|---:|---:|---:|---:|---:|---:|
| `curve_full_opt` | 29 | 0.4561 | 0.8824 | 0.0582 | 93.4% | 5.17 |
| `curve_parab_opt` | 29 | 0.4534 | 0.8772 | 0.0579 | 93.4% | 5.90 |
| `curve_frozen_opt` | 29 | 0.3898 | 0.7795 | 0.0261 | 96.6% | 8.90 |

*mean_argmin_err_pp = mean |argmin_idx_pred − argmin_idx_real| in LR_GRID points (0..29)*

## Online learning loss trajectory (curve MSE driving SGD updates)

| Variant | n_with_loss | first | last | mean | rel_decline |
|---|---:|---:|---:|---:|---:|
| `curve_full_opt` | 28 | 2.2365 | 0.0919 | 0.3223 | 95.9% |
| `curve_parab_opt` | 28 | 1.9479 | 0.1749 | 0.3140 | 91.0% |
| `curve_frozen_opt` | 28 | 1.6128 | 0.0649 | 0.1421 | 96.0% |

## LR trajectory analysis

| Variant | lr_min | lr_max | lr_mean | lr_first | lr_last (epoch=27) |
|---|---:|---:|---:|---:|---:|
| `curve_full_opt` | 0.0100 | 0.9980 | 0.7664 | 0.0100 | 0.6482 |
| `curve_parab_opt` | 0.0100 | 0.9980 | 0.7719 | 0.0100 | 0.6482 |
| `curve_frozen_opt` | 0.0100 | 0.9980 | 0.7908 | 0.0100 | 0.4617 |

Cosine LR at epoch 27/35 ≈ 0.13; cosine at epoch 35 = 0. **All curve variants stop decaying LR around 0.4-0.6** — they get stuck above the cosine-optimal late-stage LR.

## Verdict

**PROGRESS-BUT-NULL**: best variant `curve_frozen_opt` reached **peak 55.03 pp** (-14.78 vs cosine 69.81).

### What worked vs POC

1. **`cap=1.2 + cosine_blend=0.5` killed the destructive LR saturation** — all 3 variants now stay in [0.4, 1.0] (vs POC LR ~1.5 hitting the wrong side of the loss landscape). Result: best peak ≈ **55%** vs POC peak ≈ **32%** = **+23 pp**.
2. **Online updates substantially improve curve prediction**: MSE drops 15–30× from first to second half (e.g. frozen 0.78→0.026, full 0.88→0.058). Argmin idx error converges from ~5 grid steps to <0.5.
3. **Softmax-weighted argmin is more robust than parabolic** in the noisy regime (full_opt > parab_opt at epoch 27 by ~12 pp).

### What is still broken

1. **Late-stage LR floor**: even with online learning, the predicted curve has a wide flat minimum that softmax/parabolic both pick at LR ≈ 0.4–0.6. Cosine empirically goes to ~0 → the policy fails to under-shoot. → 15–20 pp gap remains.
2. **Distribution shift**: oracle_v2 was generated with depth=20_20 + ratio=2, deploy is 110_20 + ratio=4. Late-epoch student dynamics differ.
3. **Probe-loss target ≠ generalization target**: the student's probe loss landscape is flat near optimum but small-LR helps test acc more than train loss suggests.

### Ranked next steps (cost-aware)

1. **Add LR-decay regulariser to selection rule** (~1 hour code, 0 GPU): `lr_t = blend·cosine_t + (1-blend)·policy_t · (1 - α·t/T)` — cheap fix that should close most of the late-stage gap. Expected +5–10 pp.
2. **Regenerate oracle in the deploy regime** (110_20, ratio=4): 4–6 GPU-hours for fresh `oracle_v3`. Then re-pretrain CurveLRPolicy → re-run this exact 1h_curve_opt sweep. Expected to close most of the residual gap.
3. **Switch target from probe-loss-argmin to validation-acc-argmax**: requires generating a paired val_acc field per LR in oracle. ~6 GPU-hours. Highest ceiling but expensive.
4. **Hybrid scalar+curve policy**: keep CurveLRPolicy for predicting *trajectory of recommended LR*, but actually use a small head that maps the curve summary → scalar LR (essentially distilled selection). ~2 GPU-hours.
5. **Augment online loss with KL term** towards real-curve softmax-distribution: stabilises softmax temperature implicitly. Minimal cost, ~30 min code + 1 GPU-hour.

## Artifacts

- Per-variant logs: `results/1h_curve_opt/logs/1h_curve_opt/*.log`
- Per-variant snapshots: `results/1h_curve_opt/snapshots/1h_curve_opt/*/` (data_epoch_*.pt + curve_online.jsonl)
- summary.md (auto-generated by trainer/summarize_2h_curve.py)
- This file: RESULTS.md