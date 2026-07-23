#!/usr/bin/env bash
# Coffee-pull paper Wave2 (collect+train+eval). Safe to run parallel with other suites.
set -euo pipefail
cd /workspace/oat
export MUJOCO_GL=egl OAT_USE_UV_RUN=0 WANDB_MODE=disabled PYTHONUNBUFFERED=1
GPU="${GPU:-0}"
N_WORKERS="${N_WORKERS:-2}"
LOG="${LOG:-logs/awr_s10000_coffee-pull_wave2_gpu${GPU}.log}"
mkdir -p logs
echo "=== coffee-pull Wave2 START $(date -Iseconds) gpu=${GPU} workers=${N_WORKERS} ===" | tee -a "${LOG}"
SUITE=coffee-pull GPU="${GPU}" N_WORKERS="${N_WORKERS}" \
  BASE_CKPT=output/20260720/090816_train_oatpolicy_mw-coffee-pull_st_N50/checkpoints/ep-1000_sr-0.432.ckpt \
  FORCE_RECOLLECT=0 \
  TEST_START_SEED=10000 N_EXP=5 \
  bash scripts/cluster_matched_paper_wave2_awr.sh
echo "=== coffee-pull Wave2 DONE $(date -Iseconds) ===" | tee -a "${LOG}"
