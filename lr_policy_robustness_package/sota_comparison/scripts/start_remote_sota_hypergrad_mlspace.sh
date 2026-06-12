#!/usr/bin/env bash
set -uo pipefail

tmux kill-session -t sota_hypergrad_mlspace 2>/dev/null || true
cd /home/jovyan/lr-policy
mkdir -p logs/sota_hypergrad_mlspace
tmux new -d -s sota_hypergrad_mlspace \
  'cd /home/jovyan/lr-policy && REMOTE_PY=/home/jovyan/.mlspace/envs/ztf_download/bin/python bash run/run_remote_sota_hypergrad_mlspace.sh > logs/sota_hypergrad_mlspace/tmux.log 2>&1'
tmux ls
