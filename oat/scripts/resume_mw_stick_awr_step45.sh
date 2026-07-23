#!/usr/bin/env bash
# Resume stick-pull AWR after STEP1–3 already done (BoN + collect + validate).
# Usage (cluster):
#   tmux new -s mw_stick_awr_resume -d 'bash scripts/resume_mw_stick_awr_step45.sh'
set -euo pipefail
cd /workspace/oat
source .venv/bin/activate
export MUJOCO_GL=egl
export OAT_USE_UV_RUN=0
# shellcheck disable=SC1091
source scripts/cluster_gpu_env.sh 1
bash scripts/patch_robosuite_egl_assert.sh

CKPT="${CKPT:-output/20260711/134439_train_oatpolicy_mw-stick-pull_st_N50/checkpoints/ep-0800_sr-0.212.ckpt}"
AWR_DS="${AWR_DS:-my_datasets/awr_mw_stick_pull_bon.npz}"
AWR_CKPT="${AWR_CKPT:-my_models/policy_awr_mw_stick_pull.ckpt}"
AWR_OUT="${AWR_OUT:-output/eval/metaworld_stick-pull_awr_n3}"
LOG="${LOG:-logs/mw_stick_pull_awr_resume.log}"

mkdir -p logs my_models
echo "[RESUME stick AWR] $(date -Iseconds) gpu=${CUDA_VISIBLE_DEVICES}" | tee "${LOG}"

echo "[STEP4] Train AWR" | tee -a "${LOG}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  python scripts/train_awr.py \
    -i "${AWR_DS}" -c "${CKPT}" -o "${AWR_CKPT}" \
    --beta 0.5 --beta_kl 0.05 --epochs 30 --ordering uniform \
  2>&1 | tee -a "${LOG}"

echo "[STEP5] Eval AWR single-sample n=3" | tee -a "${LOG}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl \
  python scripts/eval_policy_sim.py \
    -c "${AWR_CKPT}" -o "${AWR_OUT}" -d "${OAT_DEVICE:-cuda:0}" -n 3 \
    --n_test 50 --env_task_name stick-pull \
    --use_k_tokens 8 --entropy_threshold 0 \
  2>&1 | tee -a "${LOG}"

echo "[DONE] $(date -Iseconds)" | tee -a "${LOG}"
