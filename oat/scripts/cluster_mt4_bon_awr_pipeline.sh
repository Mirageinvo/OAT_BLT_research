#!/usr/bin/env bash
# MetaWorld MT4 ep-450 (top-1) BoN→AWR pipeline — same protocol as Can/Lift:
#   STEP1 BoN eval N=8, n=3 (vote), n_test=50 (quick; full paper = chain5 250)
#   STEP2 collect 20k chunks, bon_n=8, all 4 MT4 tasks
#   STEP3 validate → STEP4 train AWR 30ep → STEP5 single-sample eval n=3
#
# Usage (cluster, GPU1):
#   tmux new -s mt4_bon_awr -d 'bash scripts/cluster_mt4_bon_awr_pipeline.sh'
set -euo pipefail
cd /workspace/oat
source .venv/bin/activate
export MUJOCO_GL=egl
export OAT_USE_UV_RUN=0
export OAT_ROOT=/workspace/oat

# shellcheck disable=SC1091
source scripts/cluster_gpu_env.sh 1
bash scripts/patch_robosuite_egl_assert.sh

CKPT="${CKPT:-output/20260708/032431_train_oatpolicy_mw-mt4_N50/checkpoints/ep-0450_sr-0.280.ckpt}"
AWR_DS="${AWR_DS:-my_datasets/awr_mt4_bon.npz}"
AWR_CKPT="${AWR_CKPT:-my_models/policy_awr_mt4.ckpt}"
BON_OUT="${BON_OUT:-output/eval/metaworld_mt4_bon_n8_n3}"
AWR_OUT="${AWR_OUT:-output/eval/metaworld_mt4_awr_n3}"
LOG="${LOG:-logs/mt4_bon_awr_pipeline.log}"

mkdir -p logs my_datasets my_models

echo "[START] $(date -Iseconds)" | tee "${LOG}"
echo "ckpt=${CKPT} gpu=${CUDA_VISIBLE_DEVICES}" | tee -a "${LOG}"

echo "[STEP1] BoN eval N=8 n=3" | tee -a "${LOG}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl \
  python scripts/eval_policy_sim.py \
    -c "${CKPT}" \
    -o "${BON_OUT}" \
    -d "${OAT_DEVICE:-cuda:0}" \
    -n 3 \
    --n_test 50 \
    --use_k_tokens 8 \
    --entropy_threshold 0 \
    --bon_free 8 \
    --bon_signal vote \
  2>&1 | tee -a "${LOG}"

echo "[STEP2] Collect AWR dataset from BoN rollouts (4 MT4 tasks)" | tee -a "${LOG}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl \
  python scripts/collect_awr_dataset.py \
    -c "${CKPT}" \
    -o "${AWR_DS}" \
    --n_chunks 20000 \
    --bon_n 8 \
    --n_tasks 4 \
    --n_workers 6 \
  2>&1 | tee -a "${LOG}"

echo "[STEP3] Validate AWR dataset" | tee -a "${LOG}"
python scripts/validate_awr.py -i "${AWR_DS}" 2>&1 | tee -a "${LOG}"

echo "[STEP4] Train AWR policy" | tee -a "${LOG}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
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
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl \
  python scripts/eval_policy_sim.py \
    -c "${AWR_CKPT}" \
    -o "${AWR_OUT}" \
    -d "${OAT_DEVICE:-cuda:0}" \
    -n 3 \
    --n_test 50 \
    --use_k_tokens 8 \
    --entropy_threshold 0 \
  2>&1 | tee -a "${LOG}"

echo "[DONE] $(date -Iseconds)" | tee -a "${LOG}"
