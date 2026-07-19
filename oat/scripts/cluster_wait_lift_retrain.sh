#!/usr/bin/env bash
# Wait for a mostly-free GPU, then start Lift retrain (seed=7, n_test=100).
set -euo pipefail
cd /workspace/oat
mkdir -p logs
LOG=logs/lift_retrain_wait_launch.log
MEM_THRESH_MIB="${MEM_THRESH_MIB:-8000}"

echo "[wait] Lift retrain queued $(date -Iseconds) thresh=${MEM_THRESH_MIB}MiB" | tee "${LOG}"

while true; do
  FREE="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', ' -v t="${MEM_THRESH_MIB}" '$2+0 < t {print $1; exit}')"
  if [[ -n "${FREE}" ]]; then
    echo "[wait] GPU ${FREE} free enough ($(date -Iseconds))" | tee -a "${LOG}"
    export GPU="${FREE}" SEED="${SEED:-7}"
    exec bash scripts/cluster_policy_lift_retrain.sh
  fi
  echo "[wait] GPUs busy; sleep 120 ($(date -Iseconds))" | tee -a "${LOG}"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tee -a "${LOG}"
  sleep 120
done
