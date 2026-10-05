#!/usr/bin/env bash
# Per-task worker: quick SR eval + one success demo (mp4+gif) with post-success hold.
set -euo pipefail
cd "$(dirname "$0")/../.."

TASK="${1:?task}"
CKPT="${2:?ckpt}"
GPU="${3:-0}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export MUJOCO_EGL_DEVICE_ID=0

OUT="${OUT:-hackathon_output}"
EVAL_DIR="${OUT}/eval/timed/${TASK}"
DEMO_DIR="${OUT}/gifs"
LOG_DIR="${OUT}/logs"
N_TEST="${N_TEST:-10}"
NUM_EXP="${NUM_EXP:-1}"
HOLD="${HOLD_AFTER_SUCCESS:-80}"
MAX_ATTEMPTS="${MAX_DEMO_ATTEMPTS:-60}"
if [[ "${OAT_USE_UV_RUN:-1}" == "0" ]]; then
  RUNNER=(python)
elif command -v uv >/dev/null 2>&1; then
  RUNNER=(uv run python)
else
  RUNNER=(python)
fi

mkdir -p "${DEMO_DIR}" "${LOG_DIR}"
log="${LOG_DIR}/eval_demo_${TASK}.log"

{
  echo "=== ${TASK} eval+demo START $(date -Iseconds) gpu=${GPU} ==="
  echo "ckpt=${CKPT}"

  echo "--- quick eval n_test=${N_TEST} ---"
  rm -rf "${EVAL_DIR}"
  "${RUNNER[@]}" scripts/eval_policy_sim.py \
    --checkpoint "${CKPT}" \
    --output_dir "${EVAL_DIR}" \
    --device "cuda:0" \
    --num_exp "${NUM_EXP}" \
    --n_test "${N_TEST}" \
    --test_start_seed 10000 \
    --n_parallel_envs 2 \
    --entropy_threshold 0 \
    --use_k_tokens 8

  echo "--- success demo (1 clip, hold=${HOLD} steps after success) ---"
  "${RUNNER[@]}" scripts/hackathon/rollout_until_success.py \
    -c "${CKPT}" \
    -o "${DEMO_DIR}" \
    --task "${TASK}" \
    --target 1 \
    --max_attempts "${MAX_ATTEMPTS}" \
    --seed_base 10000 \
    --hold_after_success "${HOLD}" \
    --use_k_tokens 8 \
    -d "cuda:0"

  echo "=== ${TASK} eval+demo DONE $(date -Iseconds) ==="
} 2>&1 | tee "${log}"
