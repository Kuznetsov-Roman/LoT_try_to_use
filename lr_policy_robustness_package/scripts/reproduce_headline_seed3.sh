#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT_DIR="${OUT_DIR:-${ARTIFACT_ROOT}/runs/headline_seed3}"
POLICY_CKPTS="${ARTIFACT_ROOT}/checkpoints/policies_beatcos/patchtst_seed0/policy.pt,${ARTIFACT_ROOT}/checkpoints/policies_beatcos/patchtst_seed2/policy.pt,${ARTIFACT_ROOT}/checkpoints/policies_beatcos/patchtst_seed5/policy.pt,${ARTIFACT_ROOT}/checkpoints/policies_beatcos/patchtst_seed7/policy.pt,${ARTIFACT_ROOT}/checkpoints/policies_beatcos/patchtst_seed4/policy.pt"
mkdir -p "${OUT_DIR}/logs" "${OUT_DIR}/snapshots" "${OUT_DIR}/ckpt"
cd "${ARTIFACT_ROOT}"
export PYTHONPATH="${ARTIFACT_ROOT}/code:${PYTHONPATH:-}"
"${PYTHON_BIN}" code/trainer/my_research.py   --exp_name patchtst_output_ensemble_top5_blend075_seed3_60ep_repro   --dataset cifar100 --datadir "${ARTIFACT_ROOT}/data/cifar" --download   --batch_size 256 --num_workers 4 --depth_list 110_20 --epochs 60   --alpha 0.5 --student_steps_ratio 4 --seed 3 --scheduler dynamic   --policy_output residual_log --policy_checkpoint "${POLICY_CKPTS}"   --policy_model_type patchtst --policy_window 10 --policy_warmup_epochs 10   --policy_min_lr 0.001 --policy_max_lr 1.5 --lr 1.0   --policy_cosine_blend 0.75 --policy_lr_ema 0.0   --snapshot_dir "${OUT_DIR}/snapshots" --save "${OUT_DIR}/ckpt/final.pt"   2>&1 | tee "${OUT_DIR}/logs/run.log"
