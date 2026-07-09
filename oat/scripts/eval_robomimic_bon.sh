#!/usr/bin/env bash
# BoN eval for RoboMimic OAT policies — same protocol as LIBERO verifier-free BoN.
# Compare against eval_robomimic_policy.sh (single-sample baseline on the SAME ckpt).
#
# Usage:
#   cd oat
#   MUJOCO_GL=egl bash scripts/eval_robomimic_bon.sh \
#     output/.../checkpoints/ep-0200_sr-0.950.ckpt lift
#
# Optional env: BON_N=8, NUM_EXP=1

set -euo pipefail

CKPT="${1:?Usage: $0 <policy.ckpt> [lift|can|square]}"
TASK="${2:-lift}"
OUT_DIR="${3:-output/eval/robomimic_${TASK}_bon}"
NUM_EXP="${NUM_EXP:-1}"
BON_N="${BON_N:-8}"

MUJOCO_GL=egl uv run scripts/eval_policy_sim.py \
  --checkpoint "${CKPT}" \
  --output_dir "${OUT_DIR}" \
  --num_exp "${NUM_EXP}" \
  --entropy_threshold 0 \
  --use_k_tokens 8 \
  --bon_free "${BON_N}" \
  --bon_signal vote
