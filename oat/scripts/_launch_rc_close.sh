#!/usr/bin/env bash
set -euo pipefail
cd /workspace/oat
export MUJOCO_GL=egl
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
# Prefer scripts/_launch_rc_pol_chain.sh (serial). Standalone waits for RAM.
NEED_KB=$(( ${NEED_MIB:-18000} * 1024 ))
while true; do
  avail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
  echo "[wait] MemAvailable=$((avail_kb/1024))MiB need>=${NEED_MIB:-18000}MiB $(date -Iseconds)"
  if [[ "${avail_kb}" -gt "${NEED_KB}" ]]; then break; fi
  sleep 120
done
SKIP_G0B=1 GPU="${GPU:-0}" N_PARALLEL_ENVS=1 NUM_WORKERS=1 TASK=close_drawer \
  bash scripts/cluster_policy_robocasa.sh
