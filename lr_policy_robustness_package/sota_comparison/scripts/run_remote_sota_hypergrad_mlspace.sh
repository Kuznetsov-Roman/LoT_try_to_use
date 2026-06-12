#!/usr/bin/env bash
# 8h SOTA comparator sweep on mlspace: batch-level hypergradient-HB.

set -uo pipefail

LOG_TOP="logs/sota_hypergrad_mlspace/top.log"
mkdir -p logs/sota_hypergrad_mlspace/per_run snapshots/sota_hypergrad_mlspace ckpt/sota_hypergrad_mlspace

REMOTE_PY="${REMOTE_PY:-/home/jovyan/.mlspace/envs/egy_llm/bin/python}"
export PATH="$(dirname "${REMOTE_PY}"):${PATH}"
export PYTHONPATH="${PYTHONPATH:-/home/jovyan/lr-policy}"

EPOCHS=60
COMMON="--dataset cifar100 --datadir data --batch_size 256 --num_workers 4 --depth_list 110_20 --epochs ${EPOCHS} --alpha 0.5 --student_steps_ratio 4 --scheduler hypergrad_hb --policy_min_lr 0.001 --policy_max_lr 1.5 --lr 1.0"

date | tee -a "${LOG_TOP}"
nvidia-smi --query-gpu=name,memory.total --format=csv | tee -a "${LOG_TOP}"

bundle_start() {
  local name="$1"
  echo "PHASE_START ${name} $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${LOG_TOP}"
  date +%s
}

bundle_end() {
  local name="$1"
  local start_ts="$2"
  local elapsed=$(( $(date +%s) - start_ts ))
  echo "PHASE_DONE ${name} elapsed_sec=${elapsed} $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${LOG_TOP}"
}

launch() {
  local exp="$1"
  local seed="$2"
  local init_lr="$3"
  local beta="$4"
  local momentum="$5"
  (
    export CUDA_VISIBLE_DEVICES=0
    ${REMOTE_PY} trainer/my_research.py \
      --exp_name "${exp}" \
      ${COMMON} \
      --seed "${seed}" \
      --hypergrad_init_lr "${init_lr}" \
      --hypergrad_hb_beta "${beta}" \
      --hypergrad_hb_momentum "${momentum}" \
      --snapshot_dir snapshots/sota_hypergrad_mlspace \
      --save "ckpt/sota_hypergrad_mlspace/${exp}.pt" \
      > "logs/sota_hypergrad_mlspace/per_run/${exp}.log" 2>&1
    echo "RUN_DONE ${exp} exit=$? $(date -u +%H:%M:%S)" >> "${LOG_TOP}"
  ) &
}

run_bundle() {
  local name="$1"
  shift
  local T
  T=$(bundle_start "${name}")
  while [[ $# -gt 0 ]]; do
    launch "$1" "$2" "$3" "$4" "$5"
    shift 5
  done
  wait
  bundle_end "${name}" "${T}"
}

run_bundle "hypergrad_hb_safe_seeds123" \
  "hypergrad_hb_safe_seed1_60ep" 1 0.50 0.020 0.90 \
  "hypergrad_hb_safe_seed2_60ep" 2 0.50 0.020 0.90 \
  "hypergrad_hb_safe_seed3_60ep" 3 0.50 0.020 0.90

run_bundle "hypergrad_hb_smooth_seeds123" \
  "hypergrad_hb_smooth_seed1_60ep" 1 0.50 0.030 0.95 \
  "hypergrad_hb_smooth_seed2_60ep" 2 0.50 0.030 0.95 \
  "hypergrad_hb_smooth_seed3_60ep" 3 0.50 0.030 0.95

run_bundle "hypergrad_hb_fast_seeds123" \
  "hypergrad_hb_fast_seed1_60ep" 1 0.30 0.050 0.90 \
  "hypergrad_hb_fast_seed2_60ep" 2 0.30 0.050 0.90 \
  "hypergrad_hb_fast_seed3_60ep" 3 0.30 0.050 0.90

echo "SOTA_HYPERGRAD_MLSPACE_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${LOG_TOP}"
