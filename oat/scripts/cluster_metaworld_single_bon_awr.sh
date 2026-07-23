#!/usr/bin/env bash
# Single-task MetaWorld BoN -> AWR pipeline (fully automated).
#
# Required:
#   TASK=box-close|coffee-pull|disassemble|stick-pull
#   CKPT=output/.../best_policy.ckpt
set -euo pipefail
cd /workspace/oat
source .venv/bin/activate

export MUJOCO_GL=egl
export OAT_USE_UV_RUN=0
export OAT_ROOT=/workspace/oat
GPU="${GPU:-0}"
# shellcheck disable=SC1091
source scripts/cluster_gpu_env.sh "${GPU}"
bash scripts/patch_robosuite_egl_assert.sh

TASK="${TASK:?Set TASK}"
CKPT="${CKPT:?Set CKPT}"
NUM_DEMO="${NUM_DEMO:-50}"
DATE_TAG="$(date +%Y%m%d)"
TIME_TAG="$(date +%H%M%S)"
RUN_TAG="mw_${TASK}_st_${DATE_TAG}_${TIME_TAG}"

AWR_DS="${AWR_DS:-my_datasets/awr_${RUN_TAG}.npz}"
AWR_CKPT="${AWR_CKPT:-my_models/policy_awr_${RUN_TAG}.ckpt}"
BON_OUT="${BON_OUT:-output/eval/metaworld_${TASK}_bon_n8_n3_${RUN_TAG}}"
AWR_OUT="${AWR_OUT:-output/eval/metaworld_${TASK}_awr_n3_${RUN_TAG}}"
LOG="${LOG:-logs/${RUN_TAG}_bon_awr.log}"

mkdir -p logs my_datasets my_models output/eval

echo "[START] $(date -Iseconds)" | tee "${LOG}"
echo "task=${TASK} ckpt=${CKPT} gpu=${CUDA_VISIBLE_DEVICES}" | tee -a "${LOG}"

if [[ ! -f "${CKPT}" ]]; then
  echo "ERROR: checkpoint not found: ${CKPT}" | tee -a "${LOG}"
  exit 1
fi
if [[ ! -d "data/metaworld/${TASK}_N${NUM_DEMO}.zarr" ]]; then
  echo "ERROR: dataset missing: data/metaworld/${TASK}_N${NUM_DEMO}.zarr" | tee -a "${LOG}"
  exit 1
fi

echo "[STEP1] BoN eval N=8 n=3 (single task)" | tee -a "${LOG}"
python scripts/eval_policy_sim.py \
  -c "${CKPT}" \
  -o "${BON_OUT}" \
  -d "${OAT_DEVICE:-cuda:0}" \
  -n 3 \
  --n_test 50 \
  --env_task_name "${TASK}" \
  --use_k_tokens 8 \
  --entropy_threshold 0 \
  --bon_free 8 \
  --bon_signal vote \
  2>&1 | tee -a "${LOG}"

echo "[STEP2] Collect AWR dataset from BoN rollouts" | tee -a "${LOG}"
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
python scripts/eval_policy_sim.py \
  -c "${AWR_CKPT}" \
  -o "${AWR_OUT}" \
  -d "${OAT_DEVICE:-cuda:0}" \
  -n 3 \
  --n_test 50 \
  --env_task_name "${TASK}" \
  --use_k_tokens 8 \
  --entropy_threshold 0 \
  2>&1 | tee -a "${LOG}"

echo "[DONE] $(date -Iseconds)" | tee -a "${LOG}"
