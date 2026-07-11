#!/usr/bin/env bash
# Resume Can BoN→AWR pipeline from STEP2 (STEP1 BoN eval already done).
# Same LIBERO-style protocol as the original launch: -n 3 BoN eval, collect bon_n=8, AWR 30ep.
#
# Usage:
#   tmux new -s can_bon_awr -d 'bash scripts/cluster_can_bon_awr_resume.sh'
set -euo pipefail
cd /workspace/oat
source .venv/bin/activate
export MUJOCO_GL=egl
export OAT_USE_UV_RUN=0
export OAT_ROOT=/workspace/oat

bash scripts/ensure_libero_config.sh
# shellcheck disable=SC1091
source scripts/cluster_gpu_env.sh 0

CKPT="${CKPT:-output/20260706/173343_train_oatpolicy_can_N200/checkpoints/ep-1700_sr-0.940.ckpt}"
AWR_DS="${AWR_DS:-my_datasets/awr_can_bon.npz}"
AWR_CKPT="${AWR_CKPT:-my_models/policy_awr_can.ckpt}"
LOG="${LOG:-logs/can_bon_awr_pipeline.log}"

mkdir -p logs my_datasets my_models

if [[ ! -f output/eval/robomimic_can_bon_n8_n3/eval_log.json ]]; then
  echo "[STEP1] BoN eval missing — run full pipeline first" | tee -a "${LOG}"
  exit 1
fi

echo "[RESUME] $(date -Iseconds) from STEP2" | tee -a "${LOG}"

echo "[STEP2] Collect AWR dataset from BoN rollouts" | tee -a "${LOG}"
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl \
  python scripts/collect_awr_dataset.py \
    -c "${CKPT}" \
    -o "${AWR_DS}" \
    --n_chunks 20000 \
    --bon_n 8 \
    --n_tasks 1 \
    --n_workers 4 \
  2>&1 | tee -a "${LOG}"

echo "[STEP3] Validate AWR dataset" | tee -a "${LOG}"
python scripts/validate_awr.py -i "${AWR_DS}" 2>&1 | tee -a "${LOG}"

echo "[STEP4] Train AWR policy" | tee -a "${LOG}"
CUDA_VISIBLE_DEVICES=0 \
  python scripts/train_awr.py \
    -i "${AWR_DS}" \
    -c "${CKPT}" \
    -o "${AWR_CKPT}" \
    --beta 0.5 \
    --beta_kl 0.05 \
    --epochs 30 \
    --ordering uniform \
  2>&1 | tee -a "${LOG}"

echo "[STEP5] Eval AWR single-sample n=3" | tee -a "${LOG}"
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl \
  python scripts/eval_policy_sim.py \
    -c "${AWR_CKPT}" \
    -o output/eval/robomimic_can_awr_n3 \
    -n 3 \
    --use_k_tokens 8 \
    --entropy_threshold 0 \
  2>&1 | tee -a "${LOG}"

echo "[DONE] $(date -Iseconds)" | tee -a "${LOG}"
