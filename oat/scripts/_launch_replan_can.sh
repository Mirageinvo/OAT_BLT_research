#!/usr/bin/env bash
# Can replan-count probe (same protocol as MW box/disassemble/stick).
# Requires: data/robomimic/hdf5_datasets/can_mh_image.hdf5
set -euo pipefail
cd /workspace/oat
source .venv/bin/activate
export MUJOCO_GL=egl
export OAT_USE_UV_RUN=0
export PYTHONUNBUFFERED=1
GPU="${GPU:-0}"
# shellcheck disable=SC1091
source scripts/cluster_gpu_env.sh "${GPU}"
bash scripts/patch_robosuite_egl_assert.sh
CKPT=output/20260706/173343_train_oatpolicy_can_N200/checkpoints/ep-1700_sr-0.940.ckpt
OUT=eval_out/replan_probe_can
LOG=logs/replan_probe_can.log
rm -rf "${OUT}"
mkdir -p logs
{
  echo "=== replan Can START $(date -Iseconds) ==="
  echo "ckpt=${CKPT}"
  echo "protocol: OAT8 -n 1 --n_test 10 --n_parallel_envs 1"
} | tee "${LOG}"
# Do NOT mkdir OUT beforehand — eval_policy_sim prompts Overwrite if the dir exists.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl \
  python scripts/eval_policy_sim.py \
    -c "${CKPT}" -o "${OUT}" -d "${OAT_DEVICE}" -n 1 \
    --use_k_tokens 8 --entropy_threshold 0 \
    --n_test 10 --n_parallel_envs 1 \
    2>&1 | tee -a "${LOG}"
echo "=== DONE replan Can $(date -Iseconds) ===" | tee -a "${LOG}"
