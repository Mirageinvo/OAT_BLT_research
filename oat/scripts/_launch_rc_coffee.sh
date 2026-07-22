#!/usr/bin/env bash
set -euo pipefail
cd /workspace/oat
export MUJOCO_GL=egl
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
# Prefer the serial chain under host-RAM pressure (SIGKILL OOM).
# Standalone: still waits for RAM, workers=0.
NEED_KB=$(( ${NEED_MIB:-18000} * 1024 ))
while true; do
  avail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
  echo "[wait] MemAvailable=$((avail_kb/1024))MiB need>=${NEED_MIB:-18000}MiB $(date -Iseconds)"
  if [[ "${avail_kb}" -gt "${NEED_KB}" ]]; then break; fi
  sleep 120
done
SKIP_G0B=1 GPU="${GPU:-1}" N_PARALLEL_ENVS=1 NUM_WORKERS=1 TASK=coffee_press_button \
  bash scripts/cluster_policy_robocasa.sh
