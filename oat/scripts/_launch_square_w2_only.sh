#!/usr/bin/env bash
# Square Wave2 only (Wave1 baseline+BoN already on disk).
set -euo pipefail
cd /workspace/oat
export MUJOCO_GL=egl OAT_USE_UV_RUN=0 WANDB_MODE=disabled PYTHONUNBUFFERED=1
GPU="${GPU:-0}"
N_WORKERS="${N_WORKERS:-2}"
SQ_BASE="${BASE_CKPT:-output/20260720/215024_train_oatpolicy_square_N200/checkpoints/ep-0700_sr-0.420.ckpt}"
LOG="${LOG:-logs/awr_s10000_square_wave2_gpu${GPU}.log}"
mkdir -p logs
# Write Wave1 summary if missing (script died after BoN last time)
if [[ ! -f output/eval/matched_s10000/square/summary.json ]]; then
  echo "=== rewrite Wave1 summary $(date -Iseconds) ===" | tee -a "${LOG}"
  SUITE=square GPU="${GPU}" \
    BASE_CKPT="${SQ_BASE}" \
    OUT_ROOT=output/eval/matched_s10000/square \
    LOG=logs/matched_s10000_square_summary_fix_gpu${GPU}.log \
    SKIP_AWR=1 SKIP_BASELINE_BON=1 FORCE_RERUN=0 \
    TEST_START_SEED=10000 N_EXP=5 \
    bash scripts/cluster_matched_triplet.sh
fi
echo "=== square Wave2 START $(date -Iseconds) gpu=${GPU} ===" | tee -a "${LOG}"
SUITE=square GPU="${GPU}" \
  BASE_CKPT="${SQ_BASE}" \
  WAVE1_ROOT=output/eval/matched_s10000/square \
  N_WORKERS="${N_WORKERS}" \
  FORCE_RECOLLECT=0 \
  TEST_START_SEED=10000 N_EXP=5 \
  bash scripts/cluster_matched_paper_wave2_awr.sh
echo "=== square Wave2 ALL DONE $(date -Iseconds) ===" | tee -a "${LOG}"
