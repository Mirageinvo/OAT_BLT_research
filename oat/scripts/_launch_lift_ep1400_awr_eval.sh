#!/usr/bin/env bash
# Lift ep1400 Wave2 AWR eval only (npz+ckpt already exist).
set -euo pipefail
cd /workspace/oat
export MUJOCO_GL=egl OAT_USE_UV_RUN=0 WANDB_MODE=disabled PYTHONUNBUFFERED=1
GPU="${GPU:-0}"
LOG="${LOG:-logs/awr_s10000_lift_ep1400_wave2_eval_gpu${GPU}.log}"
mkdir -p logs
echo "=== lift_ep1400 AWR eval START $(date -Iseconds) gpu=${GPU} ===" | tee -a "${LOG}"
SUITE=lift GPU="${GPU}" \
  BASE_CKPT=output/20260719/144024_train_oatpolicy_lift_N200/checkpoints/ep-1400_sr-0.950.ckpt \
  AWR_CKPT=my_models/awr_s10000_lift_ep1400.ckpt \
  OUT_ROOT=output/eval/matched_s10000/lift_ep1400 \
  LOG="${LOG}" \
  SKIP_AWR=0 SKIP_BASELINE_BON=1 FORCE_RERUN=1 \
  TEST_START_SEED=10000 N_EXP=5 N_PARALLEL="${N_PARALLEL:-4}" \
  bash scripts/cluster_matched_triplet.sh
echo "=== lift_ep1400 AWR eval DONE $(date -Iseconds) ===" | tee -a "${LOG}"
