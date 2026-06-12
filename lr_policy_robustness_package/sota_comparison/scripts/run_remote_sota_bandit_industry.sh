#!/usr/bin/env bash
# 8h SOTA comparator sweep on industry: non-stationary bandit over LR_GRID.

set -uo pipefail

LOG_TOP="logs/sota_bandit_industry/top.log"
mkdir -p logs/sota_bandit_industry/per_run snapshots/sota_bandit_industry ckpt/sota_bandit_industry

REMOTE_PY="${REMOTE_PY:-/home/jovyan/.mlspace/envs/egy_llm/bin/python}"
export PATH="$(dirname "${REMOTE_PY}"):${PATH}"
export PYTHONPATH="${PYTHONPATH:-/home/jovyan/lr-policy}"

EPOCHS=60
COMMON="--dataset cifar100 --datadir data --batch_size 256 --num_workers 4 --depth_list 110_20 --epochs ${EPOCHS} --alpha 0.5 --student_steps_ratio 4 --policy_min_lr 0.001 --policy_max_lr 1.5 --lr 1.0"

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
  local scheduler="$3"
  local eta="$4"
  local gamma="$5"
  local warmup="$6"
  (
    export CUDA_VISIBLE_DEVICES=0
    ${REMOTE_PY} trainer/my_research.py \
      --exp_name "${exp}" \
      ${COMMON} \
      --seed "${seed}" \
      --scheduler "${scheduler}" \
      --policy_warmup_epochs "${warmup}" \
      --bandit_eta "${eta}" --bandit_gamma "${gamma}" \
      --bandit_init_lr 0.01 \
      --snapshot_dir snapshots/sota_bandit_industry \
      --save "ckpt/sota_bandit_industry/${exp}.pt" \
      > "logs/sota_bandit_industry/per_run/${exp}.log" 2>&1
    echo "RUN_DONE ${exp} exit=$? $(date -u +%H:%M:%S)" >> "${LOG_TOP}"
  ) &
}

run_bundle() {
  local name="$1"
  shift
  local T
  T=$(bundle_start "${name}")
  while [[ $# -gt 0 ]]; do
    launch "$1" "$2" "$3" "$4" "$5" "$6"
    shift 6
  done
  wait
  bundle_end "${name}" "${T}"
}

run_bundle "bandit_exp3_safe_seeds123" \
  "bandit_exp3_safe_seed1_60ep" 1 bandit_exp3 0.05 0.10 5 \
  "bandit_exp3_safe_seed2_60ep" 2 bandit_exp3 0.05 0.10 5 \
  "bandit_exp3_safe_seed3_60ep" 3 bandit_exp3 0.05 0.10 5

run_bundle "bandit_exp3_fast_seeds123" \
  "bandit_exp3_fast_seed1_60ep" 1 bandit_exp3 0.10 0.20 3 \
  "bandit_exp3_fast_seed2_60ep" 2 bandit_exp3 0.10 0.20 3 \
  "bandit_exp3_fast_seed3_60ep" 3 bandit_exp3 0.10 0.20 3

run_bundle "bandit_ucb_seeds123" \
  "bandit_ucb_seed1_60ep" 1 bandit_ucb 0.05 0.20 5 \
  "bandit_ucb_seed2_60ep" 2 bandit_ucb 0.05 0.20 5 \
  "bandit_ucb_seed3_60ep" 3 bandit_ucb 0.05 0.20 5

echo "SOTA_BANDIT_INDUSTRY_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${LOG_TOP}"
