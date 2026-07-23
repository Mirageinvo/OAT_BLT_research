#!/usr/bin/env bash
# Queue turn_off_microwave policy when RAM frees, then exit (watcher is separate).
#
#   GPU=1 NEED_MIB=18000 bash scripts/_launch_rc_microwave_when_free.sh
set -euo pipefail
cd /workspace/oat
export MUJOCO_GL=egl WANDB_MODE=disabled PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
GPU="${GPU:-1}"
NEED_MIB="${NEED_MIB:-18000}"
NEED_KB=$((NEED_MIB * 1024))
LOG=logs/rc_microwave_queue.log
mkdir -p logs
# Prefer after coffee is locked (or never resumed) so we don't OOM-fight.
COFFEE_LOCK="${COFFEE_LOCK:-my_models/robocasa_coffee_press_button_topk_lock.txt}"
while true; do
  avail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
  coffee_busy=0
  if tmux has-session -t rc_coffee 2>/dev/null; then
    coffee_busy=1
  fi
  echo "[wait microwave] MemAvailable=$((avail_kb/1024))MiB need>=${NEED_MIB}MiB coffee_busy=${coffee_busy} lock=$([[ -f ${COFFEE_LOCK} ]] && echo y || echo n) $(date -Iseconds)" | tee -a "${LOG}"
  if [[ "${avail_kb}" -gt "${NEED_KB}" && "${coffee_busy}" -eq 0 ]]; then
    break
  fi
  sleep 120
done
SKIP_G0B=1 GPU="${GPU}" N_PARALLEL_ENVS=1 NUM_WORKERS=1 TASK=turn_off_microwave \
  bash scripts/cluster_policy_robocasa.sh
