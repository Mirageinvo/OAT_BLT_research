#!/usr/bin/env bash
# Square matched Wave1 then Wave2. Default GPU=1 (share with RC; lower n_parallel).
set -euo pipefail
cd /workspace/oat
export MUJOCO_GL=egl OAT_USE_UV_RUN=0 WANDB_MODE=disabled PYTHONUNBUFFERED=1
GPU="${GPU:-1}"
N_PARALLEL="${N_PARALLEL:-2}"
N_WORKERS="${N_WORKERS:-2}"
SQ_BASE="${BASE_CKPT:-output/20260720/215024_train_oatpolicy_square_N200/checkpoints/ep-0700_sr-0.420.ckpt}"
W1_LOG="${W1_LOG:-logs/matched_s10000_square_gpu${GPU}.log}"
mkdir -p logs
echo "=== square Wave1 START $(date -Iseconds) gpu=${GPU} n_parallel=${N_PARALLEL} ===" | tee -a "${W1_LOG}"
SUITE=square GPU="${GPU}" \
  BASE_CKPT="${SQ_BASE}" \
  OUT_ROOT=output/eval/matched_s10000/square \
  LOG="${W1_LOG}" \
  SKIP_AWR=1 FORCE_RERUN=1 \
  TEST_START_SEED=10000 N_EXP=5 N_PARALLEL="${N_PARALLEL}" \
  bash scripts/cluster_matched_triplet.sh
echo "=== square Wave1 DONE; Wave2 START $(date -Iseconds) ===" | tee -a "${W1_LOG}"
SUITE=square GPU="${GPU}" \
  BASE_CKPT="${SQ_BASE}" \
  WAVE1_ROOT=output/eval/matched_s10000/square \
  N_WORKERS="${N_WORKERS}" \
  FORCE_RECOLLECT=0 \
  TEST_START_SEED=10000 N_EXP=5 \
  bash scripts/cluster_matched_paper_wave2_awr.sh
echo "=== square ALL DONE $(date -Iseconds) ===" | tee -a "${W1_LOG}"
