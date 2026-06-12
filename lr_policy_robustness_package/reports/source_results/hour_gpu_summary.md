# Hour GPU Experiment Summary

Date: 2026-05-10

Environment:

- Conda env: `Latent`
- GPU: NVIDIA GeForce RTX 3090 Ti
- PyTorch: 2.1.2, CUDA 11.8

## Code Fixes Included

- Removed Kaggle-only CIFAR path assumptions from `utils/data.py`.
- Fixed CIFAR-10 class count and added explicit `--download`.
- Fixed local `snapshots/`, `logs/`, and `ckpt/` output paths.
- Fixed `my_research.py` test target loading to use `targets_v3_test.npy`.
- Fixed dynamic LR scheduler so `set_lr()` immediately updates optimizer param groups.
- Fixed eval metrics to use the full validation set instead of the last batch.
- Fixed the train loop to iterate over the full loader instead of repeatedly training on the first batches.
- Added `run/run_my_research_dynamic.sh` for the learnable LR-policy entrypoint.

## Experiment Setup

Common settings:

- Dataset: CIFAR-100
- Model pair: PreResNet-20 teacher, PreResNet-20 student
- Loss: `kl_ce`
- `alpha=0.5`, `T=1.5`, `student_steps_ratio=2`
- Batch size: 256
- Epochs: 35
- Seed: 0

Commands:

```bash
conda run --no-capture-output -n Latent python trainer/image_classification.py --epochs 35 --batch_size 256 --num_workers 0 --scheduler cosine --depth_list 20_20 --gpu 0 --datadir "D:\kaggle\input\datasets\kuznetsovroman\cifar-100" --exp_name hour_baseline_cosine_e35_seed0 --snapshot_dir snapshots --save ckpt/hour_gpu/hour_baseline_cosine_e35_seed0
conda run --no-capture-output -n Latent python trainer/my_research.py --epochs 35 --batch_size 256 --num_workers 0 --scheduler dynamic --policy_warmup_epochs 5 --depth_list 20_20 --gpu 0 --datadir "D:\kaggle\input\datasets\kuznetsovroman\cifar-100" --exp_name hour_dynamic_policy_e35_seed0 --snapshot_dir snapshots --save ckpt/hour_gpu/hour_dynamic_policy_e35_seed0
```

Runtime: 2760 seconds total.

## Results

| Run | Best student epoch | Best student acc | Final student acc | Final student loss | Final student F1 | Final student top-5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cosine baseline | 35 | 69.13 | 69.13 | 1.0761 | 0.6903 | 0.9174 |
| Dynamic LR policy | 30 | 62.54 | 60.23 | 1.4071 | 0.6055 | 0.8765 |

The current learnable LR policy underperforms the cosine baseline by 8.90 percentage points at the final epoch and by 6.59 percentage points at its own best epoch.

The dynamic policy predicts a narrow LR band after warmup, roughly `0.08-0.11`, while cosine keeps decaying toward zero. This likely keeps the student LR too high late in training and causes the final degradation.

## Artifacts

- Baseline metrics: `snapshots/hour_baseline_cosine_e35_seed0/metrics.jsonl`
- Dynamic policy metrics: `snapshots/hour_dynamic_policy_e35_seed0/metrics.jsonl`
- Dynamic LR trace: `snapshots/hour_dynamic_policy_e35_seed0/lr_data_epoch_*.pt`
- Baseline checkpoints: `ckpt/hour_gpu/hour_baseline_cosine_e35_seed0_teacher.pt`, `ckpt/hour_gpu/hour_baseline_cosine_e35_seed0_student.pt`
- Dynamic checkpoints: `ckpt/hour_gpu/hour_dynamic_policy_e35_seed0_teacher.pt`, `ckpt/hour_gpu/hour_dynamic_policy_e35_seed0_student.pt`

## Next Step

The immediate next experiment should change the policy output target/parameterization before more long runs:

- predict a multiplier over cosine instead of raw LR, or
- classify over the oracle LR grid and add late-epoch calibration/decay constraints.
