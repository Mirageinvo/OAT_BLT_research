#!/usr/bin/env bash
# Lift TopK-lock matched: Wave1 then Wave2 on ep-1400 (keep ep-0900 artifacts separate).
#
# Artifacts (run B / paper lock):
#   output/eval/matched_s10000/lift_ep1400/{baseline_n5,bon_n8_n5,awr_n5,summary.json}
#   my_datasets/awr_s10000_lift_ep1400.npz
#   my_models/awr_s10000_lift_ep1400.ckpt
#   logs/matched_s10000_lift_ep1400_*.log
#   logs/awr_s10000_lift_ep1400_wave2_*.log
#
# Run A (mid-train ep-0900) stays at matched_s10000/lift/ + awr_s10000_lift.*
set -euo pipefail
cd /workspace/oat

export MUJOCO_GL=egl
export OAT_USE_UV_RUN=0
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

GPU="${GPU:-0}"
BASE_CKPT="${BASE_CKPT:-output/20260719/144024_train_oatpolicy_lift_N200/checkpoints/ep-1400_sr-0.950.ckpt}"
OUT_ROOT="${OUT_ROOT:-output/eval/matched_s10000/lift_ep1400}"
WAVE1_LOG="${WAVE1_LOG:-logs/matched_s10000_lift_ep1400_gpu${GPU}.log}"
WAVE2_LOG="${WAVE2_LOG:-logs/awr_s10000_lift_ep1400_wave2_gpu${GPU}.log}"
WAVE2_EVAL_LOG="${WAVE2_EVAL_LOG:-logs/awr_s10000_lift_ep1400_wave2_eval_gpu${GPU}.log}"

[[ -f "${BASE_CKPT}" ]] || { echo "ERROR missing ${BASE_CKPT}"; exit 1; }

echo "=== lift ep1400 matched chain START $(date -Iseconds) ===" | tee -a "${WAVE1_LOG}"
echo "base=${BASE_CKPT}" | tee -a "${WAVE1_LOG}"
echo "out=${OUT_ROOT}" | tee -a "${WAVE1_LOG}"

# Wave1: baseline + BoN only
SUITE=lift GPU="${GPU}" \
  BASE_CKPT="${BASE_CKPT}" \
  OUT_ROOT="${OUT_ROOT}" \
  LOG="${WAVE1_LOG}" \
  SKIP_AWR=1 FORCE_RERUN=1 \
  TEST_START_SEED=10000 N_EXP=5 N_PARALLEL=4 \
  bash scripts/cluster_matched_triplet.sh

echo "=== Wave1 DONE $(date -Iseconds) ===" | tee -a "${WAVE1_LOG}"

# Wave2: fresh collect/train/eval under ep1400 names (do not clobber ep0900 AWR)
SUITE=lift GPU="${GPU}" \
  BASE_CKPT="${BASE_CKPT}" \
  WAVE1_ROOT="${OUT_ROOT}" \
  AWR_DS=my_datasets/awr_s10000_lift_ep1400.npz \
  AWR_CKPT=my_models/awr_s10000_lift_ep1400.ckpt \
  LOG="${WAVE2_LOG}" \
  EVAL_LOG="${WAVE2_EVAL_LOG}" \
  FORCE_RECOLLECT=1 \
  TEST_START_SEED=10000 N_EXP=5 \
  bash scripts/cluster_matched_paper_wave2_awr.sh

echo "=== lift ep1400 Wave1+Wave2 ALL DONE $(date -Iseconds) ===" | tee -a "${WAVE2_LOG}"
