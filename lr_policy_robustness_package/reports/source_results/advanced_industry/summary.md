# SOTA-architecture sweep on industry (advanced_industry)

## Setup

- Server: industry (A100-80GB).
- Phase 1 (offline pretraining, 80 epochs each): GRU baseline + 4 SOTA time-series architectures on `features_v3` (residual_log target).
- Phase 2 (online deploy, seed=20, 35 epochs CIFAR-100 110→20 distill): all 4 SOTA archs.
- Phase 3 (winner replication, seed=21, 22): best arch from Phase 2.

## Phase 1 — Offline pretraining (best test MSE, sorted)

| Arch | n_params | best test_mse | final train_loss | cfg |
|---|---:|---:|---:|---|
| `patchtst` | 6,984 | 0.2578 | 0.2652 | {'hidden': 16, 'num_layers': 2, 'max_window': 16} |
| `tcn` | 115,713 | 0.2875 | 0.0360 | {'hidden': 64, 'num_layers': 4} |
| `gru` | 349,332 | 0.3071 | 0.0555 | {} |
| `nbeats` | 1,922,679 | 0.3212 | 0.0148 | {'hidden': 128, 'num_blocks': 3, 'window': 10} |
| `dlinear` | 253 | 0.3618 | 0.3382 | {'window': 10} |

## Phase 2/3 — Deployment (final epoch 35)

| Arch | Seed | Final Student Acc | Best Student Acc | @epoch | Final Teacher Acc |
|---|---:|---:|---:|---:|---:|
| `dlinear` | 20 | 9.10 | 9.10 | 1 | 11.63 |
| `nbeats` | 20 | 9.10 | 9.10 | 1 | 12.02 |
| `patchtst` | 20 | 69.83 | 69.83 | 35 | 75.04 |
| `patchtst` | 21 | 69.70 | 69.70 | 35 | 75.40 |
| `patchtst` | 22 | 69.51 | 69.51 | 35 | 74.87 |
| `tcn` | 20 | 69.37 | 69.37 | 35 | 75.08 |

## Per-arch aggregate (final-epoch student acc)

| Arch | n_seeds | mean Final Acc | std | mean Best Acc | std |
|---|---:|---:|---:|---:|---:|
| `patchtst` | 3 | 69.68 | 0.16 | 69.68 | 0.16 |
| `tcn` | 1 | 69.37 | 0.00 | 69.37 | 0.00 |

## Reference (this codebase)

| Variant | n_seeds | mean Final Acc | source |
|---|---:|---:|---|
| `cosine` (35ep, 6 seeds) | 6 | **69.81 ± 0.33** | RESULTS_combined.md |
| `residual_log` GRU (35ep, 6 seeds) | 6 | **69.31 ± 0.43** | RESULTS_combined.md |
| `cosine` (60ep, 3 seeds, last-10 avg) | 3 | 68.42 ± 0.35 | RESULTS_full_pipeline.md |
| `residual_log` GRU (60ep, 3 seeds, last-10 avg) | 3 | 68.47 ± 0.57 | RESULTS_full_pipeline.md |

## Failures

- **NBeats** & **DLinear** crashed at deploy epoch 1 with
  `ValueError: ... expected (B,10,230), got (B,2,230)`.
  Both architectures hard-code `window=10`, but the deployment loop in
  `my_research.py` feeds variable-length feature history (only 2 features
  available at epoch 1, growing as training progresses).
  TCN and PatchTST handle variable-length windows natively, so they ran fine.
  Fix: replicate `GRULRPolicy`'s tail-padding / variable-window logic in
  `NBeatsLRPolicy` and `DLinearLRPolicy`.

