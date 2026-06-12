#!/usr/bin/env bash
# Queue hypergrad-HB on mlspace until the currently occupied GPU is available.

set -uo pipefail

cd /home/jovyan/lr-policy
mkdir -p logs/sota_hypergrad_mlspace

echo "WAITING_FOR_FREE_GPU $(date -u +%Y-%m-%dT%H:%M:%SZ)"
while true; do
  mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'NR==1 {print $1}')
  if [[ "${mem_used}" -le 5000 ]]; then
    break
  fi
  echo "GPU_BUSY mem=${mem_used} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sleep 300
done

REMOTE_PY=/home/jovyan/.mlspace/envs/ztf_download/bin/python \
  bash run/run_remote_sota_hypergrad_mlspace.sh
