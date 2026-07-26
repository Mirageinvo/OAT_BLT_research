#!/usr/bin/env bash
set -euo pipefail
cd /workspace/oat
UVPY=/workspace/oat/_runtime_uv_python/cpython-3.10.20-linux-x86_64-gnu/bin/python3.10
export VIRTUAL_ENV=/workspace/oat/.venv
export PATH="/workspace/oat/.venv/bin:$PATH"
export PYTHONPATH="/workspace/oat/.venv/lib/python3.10/site-packages:${PYTHONPATH:-}"
export OAT_USE_UV_RUN=0
export MUJOCO_GL=egl

SUITE="${1:?suite}"
GPU="${2:?gpu}"
CKPT="${3:?ckpt}"

source scripts/cluster_gpu_env.sh "${GPU}"
# patch with UVPY
VENV=/workspace/oat/.venv bash scripts/patch_robosuite_egl_assert.sh || true

OUT_ROOT="output/eval/matched_s10000/${SUITE}"
LOG="logs/mw_${SUITE}_bon32_only_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs "${OUT_ROOT}"
OUT="${OUT_ROOT}/bon_n32_n5"
rm -rf "${OUT}"

echo "=== MW ${SUITE} BoN32-only | gpu=${GPU} | $(date -Iseconds) ===" | tee "${LOG}"
echo "ckpt=${CKPT} py=${UVPY}" | tee -a "${LOG}"
"$UVPY" -c "import torch,hydra; print(\"cuda\", torch.cuda.is_available(), torch.cuda.device_count()); print(\"hydra\", hydra.__file__)" | tee -a "${LOG}"
test -f "${CKPT}"
test -x "${UVPY}"

echo "[$(date -Iseconds)] START BoN N=32 -> ${OUT}" | tee -a "${LOG}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl \
  "$UVPY" scripts/eval_policy_sim.py \
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
    --bon_free 32 \
    --bon_signal vote \
  2>&1 | tee -a "${LOG}"
echo "[$(date -Iseconds)] DONE BoN N=32" | tee -a "${LOG}"
echo "=== CHAIN DONE $(date -Iseconds) ===" | tee -a "${LOG}"
