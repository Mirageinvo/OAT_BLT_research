#!/usr/bin/env bash
# Launch one RoboCasa policy with optional RAM wait.
#   TASK=close_drawer GPU=0 NEED_MIB=12000 bash scripts/_launch_rc_policy.sh
set -euo pipefail
cd /workspace/oat
export MUJOCO_GL=egl WANDB_MODE=disabled PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
TASK="${TASK:?Set TASK}"
GPU="${GPU:-0}"
NEED_MIB="${NEED_MIB:-12000}"
NEED_KB=$((NEED_MIB * 1024))
while true; do
  avail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
  echo "[wait ${TASK}] MemAvailable=$((avail_kb/1024))MiB need>=${NEED_MIB}MiB $(date -Iseconds)"
  if [[ "${avail_kb}" -gt "${NEED_KB}" ]]; then break; fi
  sleep 90
done
SKIP_G0B=1 GPU="${GPU}" N_PARALLEL_ENVS=1 NUM_WORKERS="${NUM_WORKERS:-1}" TASK="${TASK}" \
  bash scripts/cluster_policy_robocasa.sh
