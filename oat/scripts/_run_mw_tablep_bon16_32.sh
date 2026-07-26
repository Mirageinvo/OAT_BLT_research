#!/usr/bin/env bash
# One Table-P MW task: BoN16 then BoN32 (matched_s10000, n=5).
set -euo pipefail
cd /workspace/oat
source .venv/bin/activate
export OAT_USE_UV_RUN=0
export MUJOCO_GL=egl

SUITE="${1:?suite}"
GPU="${2:?gpu}"
CKPT="${3:?ckpt}"

source scripts/cluster_gpu_env.sh "${GPU}"
bash scripts/patch_robosuite_egl_assert.sh || true

OUT_ROOT="output/eval/matched_s10000/${SUITE}"
LOG="logs/mw_${SUITE}_bon16_32_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs "${OUT_ROOT}"

echo "=== MW ${SUITE} BoN16→32 | gpu=${GPU} | $(date -Iseconds) ===" | tee "${LOG}"
echo "ckpt=${CKPT}" | tee -a "${LOG}"
test -f "${CKPT}"

run_bon() {
  local N="$1"
  local OUT="${OUT_ROOT}/bon_n${N}_n5"
  rm -rf "${OUT}"
  echo "" | tee -a "${LOG}"
  echo "[$(date -Iseconds)] START BoN N=${N} -> ${OUT}" | tee -a "${LOG}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl \
    python scripts/eval_policy_sim.py \
      -c "${CKPT}" \
      -o "${OUT}" \
      -d "${OAT_DEVICE}" \
      -n 5 \
      --n_test 50 \
      --test_start_seed 10000 \
      --n_parallel_envs 2 \
      --use_k_tokens 8 \
      --entropy_threshold 0 \
      --temperature 1.0 \
      --topk 10 \
      --env_task_name "${SUITE}" \
      --bon_free "${N}" \
      --bon_signal vote \
    2>&1 | tee -a "${LOG}"
  echo "[$(date -Iseconds)] DONE BoN N=${N}" | tee -a "${LOG}"
}

run_bon 16
run_bon 32
echo "=== CHAIN DONE $(date -Iseconds) ===" | tee -a "${LOG}"
