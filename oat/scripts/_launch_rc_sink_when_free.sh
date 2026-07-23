#!/usr/bin/env bash
# Queue turn_off_sink_faucet when RAM frees and coffee is not occupying the slot.
#   NEED_MIB=20000 GPU=1 bash scripts/_launch_rc_sink_when_free.sh
set -euo pipefail
cd /workspace/oat
export MUJOCO_GL=egl WANDB_MODE=disabled PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
GPU="${GPU:-1}"
NEED_MIB="${NEED_MIB:-20000}"
NEED_KB=$((NEED_MIB * 1024))
LOG=logs/rc_sink_queue.log
mkdir -p logs
while true; do
  avail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
  coffee_busy=0
  if pgrep -af "task/policy=robocasa/coffee_press_button" >/dev/null 2>&1; then
    coffee_busy=1
  fi
  echo "[wait sink] MemAvailable=$((avail_kb/1024))MiB need>=${NEED_MIB}MiB coffee_busy=${coffee_busy} $(date -Iseconds)" | tee -a "${LOG}"
  if [[ "${avail_kb}" -gt "${NEED_KB}" && "${coffee_busy}" -eq 0 ]]; then
    break
  fi
  sleep 120
done
# Fresh start: skip epoch-0 eval (rollout_start_epoch=100 via cluster_policy_robocasa.sh)
SKIP_G0B=1 GPU="${GPU}" N_PARALLEL_ENVS=1 NUM_WORKERS=1 TASK=turn_off_sink_faucet \
  bash scripts/cluster_policy_robocasa.sh
