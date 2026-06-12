#!/usr/bin/env bash
# 8h SOTA comparator sweep on stars: AdaLRS / GreedyLR-style probe search.

set -uo pipefail

LOG_TOP="logs/sota_adalrs_stars/top.log"
mkdir -p logs/sota_adalrs_stars/per_run snapshots/sota_adalrs_stars ckpt/sota_adalrs_stars

REMOTE_PY="${REMOTE_PY:-/home/jovyan/.mlspace/envs/egy_llm/bin/python}"
export PATH="$(dirname "${REMOTE_PY}"):${PATH}"
export PYTHONPATH="${PYTHONPATH:-/home/jovyan/lr-policy}"

EPOCHS=60
COMMON="--dataset cifar100 --datadir data --batch_size 256 --num_workers 4 --depth_list 110_20 --epochs ${EPOCHS} --alpha 0.5 --student_steps_ratio 4 --scheduler adalrs --policy_min_lr 0.001 --policy_max_lr 1.5 --lr 1.0"

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
  local alpha="$3"
  local beta="$4"
  local margin="$5"
  local clamp="$6"
  local warmup="$7"
  (
    export CUDA_VISIBLE_DEVICES=0
    ${REMOTE_PY} trainer/my_research.py \
      --exp_name "${exp}" \
      ${COMMON} \
      --seed "${seed}" \
      --policy_warmup_epochs "${warmup}" \
      --adalrs_alpha "${alpha}" --adalrs_beta "${beta}" \
      --adalrs_margin "${margin}" --adalrs_clamp "${clamp}" \
      --adalrs_init_lr 0.01 \
      --snapshot_dir snapshots/sota_adalrs_stars \
      --save "ckpt/sota_adalrs_stars/${exp}.pt" \
      > "logs/sota_adalrs_stars/per_run/${exp}.log" 2>&1
    echo "RUN_DONE ${exp} exit=$? $(date -u +%H:%M:%S)" >> "${LOG_TOP}"
  ) &
}

run_bundle() {
  local name="$1"
  shift
  local T
  T=$(bundle_start "${name}")
  while [[ $# -gt 0 ]]; do
    launch "$1" "$2" "$3" "$4" "$5" "$6" "$7"
    shift 7
  done
  wait
  bundle_end "${name}" "${T}"
}

run_bundle "adalrs_default_seeds123" \
  "adalrs_default_seed1_60ep" 1 0.50 1.50 0.0020 0.70 5 \
  "adalrs_default_seed2_60ep" 2 0.50 1.50 0.0020 0.70 5 \
  "adalrs_default_seed3_60ep" 3 0.50 1.50 0.0020 0.70 5

run_bundle "adalrs_narrow_safe_seeds123" \
  "adalrs_narrow_safe_seed1_60ep" 1 0.70 1.30 0.0010 0.80 5 \
  "adalrs_narrow_safe_seed2_60ep" 2 0.70 1.30 0.0010 0.80 5 \
  "adalrs_narrow_safe_seed3_60ep" 3 0.70 1.30 0.0010 0.80 5

run_bundle "adalrs_aggressive_seeds123" \
  "adalrs_aggressive_seed1_60ep" 1 0.40 1.80 0.0005 0.90 3 \
  "adalrs_aggressive_seed2_60ep" 2 0.40 1.80 0.0005 0.90 3 \
  "adalrs_aggressive_seed3_60ep" 3 0.40 1.80 0.0005 0.90 3

echo "SOTA_ADALRS_STARS_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${LOG_TOP}"
