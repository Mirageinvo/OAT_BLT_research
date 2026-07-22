#!/usr/bin/env bash
# 06 — train the OAT policy (frozen tokenizer from 05). Checkpoints selected by sim SR
# (needs the filled RoboTwinRunner). Output -> your policy checkpoint.
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
echo "[06] train policy (task/policy=robotwin/${TASK_CFG})"
uv run accelerate launch scripts/run_workspace.py \
  --config-name=train_oatpolicy task/policy=robotwin/${TASK_CFG}
echo "[06] done. Set POLICY_CKPT in config.sh to the best saved policy checkpoint, then run 07."
