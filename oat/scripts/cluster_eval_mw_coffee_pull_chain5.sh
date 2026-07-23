#!/usr/bin/env bash
# Paper-style chain5 eval for MetaWorld coffee-pull single-task policy.
# 5 eval seeds × 50 rollouts = 250 episodes (same protocol as Lift/Can chain5).
#
# Usage (cluster):
#   tmux new -s mwst_chain5_coffee_pull -d 'bash scripts/cluster_eval_mw_coffee_pull_chain5.sh'
set -euo pipefail
cd /workspace/oat
source .venv/bin/activate

export OAT_USE_UV_RUN=0
export MUJOCO_GL=egl
GPU="${GPU:-1}"

CKPT="${CKPT:-output/20260711/134440_train_oatpolicy_mw-coffee-pull_st_N50/checkpoints/ep-1000_sr-0.432.ckpt}"
OUT_ROOT="${OUT_ROOT:-output/eval/metaworld_coffee-pull_paper5_ep1000}"
LOG="${LOG:-logs/eval_mw_coffee_pull_chain5.log}"

TASK=coffee-pull CKPT="${CKPT}" GPU="${GPU}" OUT_ROOT="${OUT_ROOT}" LOG="${LOG}" \
  N_TEST="${N_TEST:-50}" BASE_SEED="${BASE_SEED:-1000}" N_PARALLEL="${N_PARALLEL:-4}" \
  bash scripts/cluster_eval_metaworld_single_chain5.sh
