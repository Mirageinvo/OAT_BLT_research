#!/usr/bin/env bash
# Paper-style chain5 eval for MetaWorld stick-pull single-task policy.
# 5 eval seeds × 50 rollouts = 250 episodes (same protocol as Lift/Can chain5).
#
# Usage (cluster):
#   tmux new -s mwst_chain5_stick_pull -d 'bash scripts/cluster_eval_mw_stick_pull_chain5.sh'
set -euo pipefail
cd /workspace/oat
source .venv/bin/activate

export OAT_USE_UV_RUN=0
export MUJOCO_GL=egl
GPU="${GPU:-0}"

CKPT="${CKPT:-output/20260711/134439_train_oatpolicy_mw-stick-pull_st_N50/checkpoints/ep-0800_sr-0.212.ckpt}"
OUT_ROOT="${OUT_ROOT:-output/eval/metaworld_stick-pull_paper5_ep0800}"
LOG="${LOG:-logs/eval_mw_stick_pull_chain5.log}"

TASK=stick-pull CKPT="${CKPT}" GPU="${GPU}" OUT_ROOT="${OUT_ROOT}" LOG="${LOG}" \
  N_TEST="${N_TEST:-50}" BASE_SEED="${BASE_SEED:-1000}" N_PARALLEL="${N_PARALLEL:-4}" \
  bash scripts/cluster_eval_metaworld_single_chain5.sh
