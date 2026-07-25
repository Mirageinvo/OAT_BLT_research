#!/usr/bin/env bash
# 06 — train the OAT policy (frozen tokenizer from 05). Checkpoints selected by sim SR
# (needs the filled RoboTwinRunner). Output -> your policy checkpoint.
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
# IN-TRAINING sim eval DISABLED for RoboTwin (rollout_every > num_epochs): our AsyncVectorEnv
# runner forks, which SAPIEN/Vulkan doesn't survive. Checkpoints still save every checkpoint_every
# epochs (by loss/step). Sim eval is done SEPARATELY via the RoboTwin harness (path B): 10_eval_pathB.sh.
: "${TOK_CKPT:?set TOK_CKPT in config.sh to the trained tokenizer checkpoint (05) first}"
echo "[06] train policy (task/policy=robotwin/${TASK_CFG})  num_demo=${NDEMO}  tokenizer=${TOK_CKPT}  (sim-eval OFF)"
# rollout OFF (fork/SAPIEN) -> switch topk checkpoint selection from mean_success_rate to val_loss
${OATACCEL} scripts/run_workspace.py \
  --config-name=train_oatpolicy task/policy=robotwin/${TASK_CFG} \
  training.num_demo=${NDEMO} training.rollout_every=99999999 \
  policy.action_tokenizer.checkpoint="${TOK_CKPT}" \
  checkpoint.topk.monitor_key=val_loss checkpoint.topk.mode=min \
  'checkpoint.topk.format_str="ep-{epoch:04d}_vl-{val_loss:.4f}.ckpt"' \
  "$@"   # extra Hydra overrides; to RESUME pass hydra.run.dir=<original run dir> (else new empty dir -> starts fresh)
echo "[06] done. Find the latest checkpoint under output/<date>/... ; set POLICY_CKPT in config.sh,"
echo "     then eval via path B: bash robotwin_pipeline/10_eval_pathB.sh"
