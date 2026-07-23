#!/usr/bin/env bash
# Resume RoboCasa coffee_press_button policy after SIGKILL, then hand off to plateau watcher.
#
#   GPU=1 NEED_MIB=18000 bash scripts/_launch_rc_coffee_resume.sh
set -euo pipefail
cd /workspace/oat

export MUJOCO_GL=egl
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
export OAT_USE_UV_RUN=0

VENV="${VENV:-/workspace/oat/.venv_robocasa}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

RUN_DIR="${RUN_DIR:-output/20260721/204916_train_oatpolicy_coffee_press_button_N200}"
TOKENIZER_CKPT="${TOKENIZER_CKPT:-output/20260720/041753_train_oattok_coffee_press_button_N200/checkpoints/ep-1940_mse-0.003.ckpt}"
GPU="${GPU:-1}"
NEED_MIB="${NEED_MIB:-18000}"
NEED_KB=$((NEED_MIB * 1024))
LOG="${LOG:-logs/train_policy_robocasa_coffee_press_button_s0_resume.log}"
mkdir -p logs

while true; do
  avail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
  echo "[wait coffee_resume] MemAvailable=$((avail_kb/1024))MiB need>=${NEED_MIB}MiB $(date -Iseconds)" | tee -a "${LOG}"
  if [[ "${avail_kb}" -gt "${NEED_KB}" ]]; then
    break
  fi
  sleep 120
done

# shellcheck disable=SC1091
source scripts/cluster_gpu_env.sh "${GPU}"
VENV="${VENV}" bash scripts/patch_robosuite_egl_assert.sh || true

echo "=== coffee_press resume START $(date -Iseconds) gpu=${GPU} run=${RUN_DIR} ===" | tee -a "${LOG}"
SKIP_G0B=1 HYDRA_FULL_ERROR=1 accelerate launch \
  --num_machines 1 --num_processes 1 \
  scripts/run_workspace.py \
  --config-name=train_oatpolicy \
  task/policy=robocasa/coffee_press_button \
  task.policy.lazy_eval=false \
  task.policy.env_runner.n_test=50 \
  task.policy.env_runner.n_parallel_envs=1 \
  task.policy.env_runner.test_start_seed=2000 \
  "policy.action_tokenizer.checkpoint=${TOKENIZER_CKPT}" \
  training.num_demo=200 \
  training.rollout_every=100 \
  training.seed=0 \
  seed=0 \
  dataloader.num_workers=1 \
  val_dataloader.num_workers=1 \
  checkpoint.topk.k=3 \
  checkpoint.topk.monitor_key=mean_success_rate \
  logging.mode=disabled \
  "hydra.run.dir=${RUN_DIR}" \
  training.resume=true \
  2>&1 | tee -a "${LOG}"

echo "=== coffee_press resume finished $(date -Iseconds) ===" | tee -a "${LOG}"
