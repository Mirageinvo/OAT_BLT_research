#!/usr/bin/env bash
# One-shot full eval for the best trained RoboMimic Lift policy checkpoint.
set -euo pipefail
cd /workspace/oat
source .venv/bin/activate

export OAT_USE_UV_RUN=0
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"

CKPT="${CKPT:-output/20260706/163500_train_oatpolicy_lift_N200/checkpoints/ep-0600_sr-0.920.ckpt}"
OUT="${OUT:-output/eval/robomimic_lift_best_ep0600_n250}"
LOG="${LOG:-logs/eval_lift_best_ep0600_n250.log}"

mkdir -p logs "$(dirname "${OUT}")"
echo "=== Lift best checkpoint full eval | n_test=250 | OAT8 | $(date -Iseconds) ===" | tee "${LOG}"
echo "ckpt=${CKPT}" | tee -a "${LOG}"
echo "out=${OUT}" | tee -a "${LOG}"
echo "gpu=${CUDA_VISIBLE_DEVICES}" | tee -a "${LOG}"

python scripts/eval_policy_sim.py \
  -c "${CKPT}" \
  -o "${OUT}" \
  --n_test 250 \
  --n_parallel_envs 2 \
  --use_k_tokens 8 \
  --entropy_threshold 0 \
  2>&1 | tee -a "${LOG}"
