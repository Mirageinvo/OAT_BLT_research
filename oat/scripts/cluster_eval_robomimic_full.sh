#!/usr/bin/env bash
# Full paper-style eval for a RoboMimic OAT policy checkpoint.
#
# Paper Table VI protocol note:
#   true paper = 5 *training* seeds × 50 rollouts each.
#   With a single trained seed, NUM_EXP=5 repeats the 50-rollout eval
#   (same pattern as scripts/eval_metaworld_policy.sh) → mean ± stderr.
#
# Usage:
#   OAT_USE_UV_RUN=0 bash scripts/cluster_eval_robomimic_full.sh \
#     output/.../ep-0600_sr-0.920.ckpt lift
set -euo pipefail
cd /workspace/oat
source .venv/bin/activate

export OAT_USE_UV_RUN="${OAT_USE_UV_RUN:-0}"
export MUJOCO_GL=egl
export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"

CKPT="${1:?Usage: $0 <policy.ckpt> [lift|can|square]}"
TASK="${2:-lift}"
NUM_EXP="${NUM_EXP:-5}"
OUT_DIR="${3:-output/eval/robomimic_${TASK}_full_n${NUM_EXP}}"
LOG="logs/eval_${TASK}_full_n${NUM_EXP}.log"

# Prefer free GPU; default GPU0 (shared with can/square). Override with CUDA_VISIBLE_DEVICES.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_EGL_DEVICE_ID=0

mkdir -p logs "$(dirname "${OUT_DIR}")"
echo "=== FULL EVAL | task=${TASK} | NUM_EXP=${NUM_EXP} | GPU=${CUDA_VISIBLE_DEVICES} ===" | tee "${LOG}"
echo "    ckpt=${CKPT}" | tee -a "${LOG}"
echo "    out=${OUT_DIR}" | tee -a "${LOG}"
echo "    flags: --use_k_tokens 8 --entropy_threshold 0" | tee -a "${LOG}"

if [[ "${OAT_USE_UV_RUN}" == "0" ]]; then
  python scripts/eval_policy_sim.py \
    --checkpoint "${CKPT}" \
    --output_dir "${OUT_DIR}" \
    --num_exp "${NUM_EXP}" \
    --entropy_threshold 0 \
    --use_k_tokens 8 \
    2>&1 | tee -a "${LOG}"
else
  MUJOCO_GL=egl uv run scripts/eval_policy_sim.py \
    --checkpoint "${CKPT}" \
    --output_dir "${OUT_DIR}" \
    --num_exp "${NUM_EXP}" \
    --entropy_threshold 0 \
    --use_k_tokens 8 \
    2>&1 | tee -a "${LOG}"
fi
