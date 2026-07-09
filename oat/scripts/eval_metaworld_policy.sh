#!/usr/bin/env bash
# Eval baseline (OAT8, single sample) on MetaWorld MT4.
#
# Usage:
#   MUJOCO_GL=egl bash scripts/eval_metaworld_policy.sh output/.../ep-xxxx_sr-0.244.ckpt mt4

set -euo pipefail

CKPT="${1:?Usage: $0 <policy.ckpt> [mt4]}"
TASK="${2:-mt4}"
OUT_DIR="${3:-output/eval/metaworld_${TASK}}"
NUM_EXP="${NUM_EXP:-5}"

MUJOCO_GL=egl uv run scripts/eval_policy_sim.py \
  --checkpoint "${CKPT}" \
  --output_dir "${OUT_DIR}" \
  --num_exp "${NUM_EXP}" \
  --entropy_threshold 0 \
  --use_k_tokens 8
