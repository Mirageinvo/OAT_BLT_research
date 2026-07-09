#!/usr/bin/env bash
# Eval an AWR-distilled RoboMimic policy (single-sample, no BoN at inference).
#
# Usage:
#   cd oat
#   MUJOCO_GL=egl bash scripts/eval_robomimic_awr.sh my_models/policy_awr_lift.ckpt lift

set -euo pipefail

CKPT="${1:?Usage: $0 <awr_policy.ckpt> [lift|can|square]}"
TASK="${2:-lift}"
OUT_DIR="${3:-output/eval/robomimic_${TASK}_awr}"
NUM_EXP="${NUM_EXP:-1}"

MUJOCO_GL=egl uv run scripts/eval_policy_sim.py \
  --checkpoint "${CKPT}" \
  --output_dir "${OUT_DIR}" \
  --num_exp "${NUM_EXP}" \
  --entropy_threshold 0 \
  --use_k_tokens 8
