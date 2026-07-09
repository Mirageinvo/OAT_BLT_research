#!/usr/bin/env bash
# Verifier-free BoN eval on MetaWorld MT4.
#
# Usage:
#   MUJOCO_GL=egl bash scripts/eval_metaworld_bon.sh output/.../policy.ckpt mt4

set -euo pipefail

CKPT="${1:?Usage: $0 <policy.ckpt> [mt4]}"
TASK="${2:-mt4}"
OUT_DIR="${3:-output/eval/metaworld_${TASK}_bon}"
NUM_EXP="${NUM_EXP:-5}"
BON_N="${BON_N:-8}"

MUJOCO_GL=egl uv run scripts/eval_policy_sim.py \
  --checkpoint "${CKPT}" \
  --output_dir "${OUT_DIR}" \
  --num_exp "${NUM_EXP}" \
  --entropy_threshold 0 \
  --use_k_tokens 8 \
  --bon_free "${BON_N}" \
  --bon_signal vote
