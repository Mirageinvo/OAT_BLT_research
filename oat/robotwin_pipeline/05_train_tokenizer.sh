#!/usr/bin/env bash
# 05 — train the OAT tokenizer (action-only). CHECK the printed recon-MSE at the end.
# If MSE >> LIBERO's ~0.002, re-run with: NUM_REGISTERS=16 bash 05_train_tokenizer.sh
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
EXTRA="training.num_demo=${NDEMO}"
if [ -n "${NUM_REGISTERS:-}" ]; then EXTRA="${EXTRA} tokenizer.encoder.num_registers=${NUM_REGISTERS}"; fi
echo "[05] train tokenizer (task/tokenizer=robotwin/${TASK_CFG}) ${EXTRA}"
uv run accelerate launch scripts/run_workspace.py \
  --config-name=train_oattok task/tokenizer=robotwin/${TASK_CFG} ${EXTRA}
echo "[05] done. Set TOK_CKPT in config.sh to the saved tokenizer checkpoint, then run 06."
echo "     If recon-MSE was poor -> re-run with NUM_REGISTERS=16 (and use that policy config too)."
