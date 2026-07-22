#!/usr/bin/env bash
# Serial RoboCasa policy restart after SIGKILL/OOM.
# No ckpts were saved (died mid epoch-0 eval) → fresh runs, not hydra resume.
#
# OOM fixes vs previous launch:
#   - ONE task at a time (never coffee∥close)
#   - NUM_WORKERS=0 (no dataloader fork RAM)
#   - N_PARALLEL_ENVS=1
#   - wait MemAvailable >= NEED_MIB before each task
#
# Usage (cluster docker):
#   bash scripts/_launch_rc_pol_chain.sh
set -euo pipefail
cd /workspace/oat

export MUJOCO_GL=egl
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"

NEED_MIB="${NEED_MIB:-18000}"   # ~18 GiB MemAvailable
NEED_KB=$((NEED_MIB * 1024))
WAIT_LOG=logs/rc_pol_chain_wait.log
mkdir -p logs

wait_ram() {
  local tag="$1"
  while true; do
    avail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
    avail_mib=$((avail_kb / 1024))
    echo "[wait ${tag}] MemAvailable=${avail_mib}MiB need>=${NEED_MIB}MiB $(date -Iseconds)" | tee -a "${WAIT_LOG}"
    if [[ "${avail_kb}" -gt "${NEED_KB}" ]]; then
      echo "[wait ${tag}] OK — starting $(date -Iseconds)" | tee -a "${WAIT_LOG}"
      break
    fi
    sleep 120
  done
}

echo "=== rc_pol_chain START $(date -Iseconds) ===" | tee -a "${WAIT_LOG}"

wait_ram coffee
# GPU1 usually quieter (square/lift on 0). auto still ok if pinned busy.
# NUM_WORKERS must be >=1 if DataLoader persistent_workers=True (OAT default).
SKIP_G0B=1 GPU="${COFFEE_GPU:-1}" N_PARALLEL_ENVS=1 NUM_WORKERS=1 \
  TASK=coffee_press_button \
  bash scripts/cluster_policy_robocasa.sh
echo "=== coffee_press_button DONE $(date -Iseconds) ===" | tee -a "${WAIT_LOG}"

wait_ram close
SKIP_G0B=1 GPU="${CLOSE_GPU:-0}" N_PARALLEL_ENVS=1 NUM_WORKERS=1 \
  TASK=close_drawer \
  bash scripts/cluster_policy_robocasa.sh
echo "=== close_drawer DONE $(date -Iseconds) ===" | tee -a "${WAIT_LOG}"

echo "=== rc_pol_chain ALL DONE $(date -Iseconds) ===" | tee -a "${WAIT_LOG}"
