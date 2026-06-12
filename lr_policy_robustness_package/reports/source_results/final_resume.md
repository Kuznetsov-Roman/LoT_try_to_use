# Final Resume: Learnable LR Policy Starting Point

Date: 2026-05-10

## Executive Summary

The repository is now technically runnable for the CIFAR-100 learnable LR-policy experiment. The original raw LR policy was reproduced and underperformed the cosine baseline badly. A safer policy parameterization, `cosine_multiplier`, was added and evaluated on `stars` with 6 parallel seeds on an A100.

Final result: the safe multiplier policy reaches baseline-level performance, but does not yet provide a clear improvement.

## What Was Fixed

- Removed Kaggle-only CIFAR path assumptions from `utils/data.py`.
- Added explicit `--download` support for torchvision datasets.
- Fixed CIFAR-10 class count from 100 to 10.
- Removed an unused `matplotlib` import that broke the remote `stars` environment.
- Fixed `my_research.py` to load `targets_v3_test.npy` for test evaluation.
- Fixed `DynamicScheduler.set_lr()` so it immediately updates optimizer param groups.
- Fixed validation metrics to use the full test set, not the last batch.
- Reworked the train loop to iterate over the real dataloader instead of repeatedly using the first batches.
- Added local `snapshots/`, `logs/`, and `ckpt/` output paths.
- Added `cosine_multiplier` policy output mode.
- Added remote bundle launcher scripts for high A100 utilization without increasing batch size.

## Experiments

| Run | Location | Seeds | Final student acc | Best student acc | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| Cosine baseline | local RTX 3090 Ti | 1 | 69.13 | 69.13 | `image_classification.py`, 35 epochs |
| Raw dynamic LR policy | local RTX 3090 Ti | 1 | 60.23 | 62.54 | predicts raw LR around `0.08-0.11`, too high late |
| Safe cosine multiplier policy | `stars` A100 | 6 | 69.17 ± 0.14 | 69.19 mean | batch stayed 256, ran 6 seeds in parallel |

Safe multiplier per-seed final student accuracy:

| Seed | Final acc | Best acc | Best epoch | Final loss |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 69.00 | 69.00 | 35 | 1.0909 |
| 1 | 69.25 | 69.25 | 35 | 1.0590 |
| 2 | 69.38 | 69.38 | 35 | 1.0541 |
| 3 | 69.07 | 69.07 | 35 | 1.0875 |
| 4 | 69.10 | 69.10 | 35 | 1.0673 |
| 5 | 69.23 | 69.32 | 34 | 1.0669 |

## Interpretation

The original learnable policy is not useful as-is: direct raw LR prediction destabilizes or slows the student compared with cosine. The safe multiplier policy fixes the worst failure mode by constraining the predicted LR relative to cosine, and it recovers baseline performance across seeds.

This means the current code is a valid starting point for research, but not yet a positive result. The policy features contain signal, but the current supervised target and deployment rule are not enough to beat a strong hand-designed schedule.

## Artifacts

Local:

- `results/hour_gpu_summary.md`
- `results/remote_safe_multiplier/`
- `snapshots/hour_baseline_cosine_e35_seed0/metrics.jsonl`
- `snapshots/hour_dynamic_policy_e35_seed0/metrics.jsonl`

Remote `stars`:

- Repo sandbox: `/home/jovyan/lr-policy`
- Python: `/home/jovyan/.mlspace/envs/egy_llm/bin/python`
- Bundle log: `/home/jovyan/lr-policy/logs/remote/bundle_safe_multiplier.log`
- Metrics: `/home/jovyan/lr-policy/snapshots/remote_safe_multiplier_policy_e35_seed*/metrics.jsonl`

## Recommended Next Step

Do not spend more GPU on the same raw or safe multiplier setup. The next research step should change the learning objective:

1. Train the policy as classification over the oracle LR grid instead of scalar regression.
2. Predict either a discrete cosine multiplier bucket or a residual over cosine.
3. Add a validation-time rule that prevents late-epoch LR increases unless the landscape probe shows a clear improvement.

The best immediate experiment is a discrete policy head over the existing oracle grid, evaluated with the same 6-seed A100 bundle.
