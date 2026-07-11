#!/usr/bin/env bash
# Launch 4 parallel single-task MetaWorld POLICY runs (tokenizer already trained).
# Paper-style train-time eval: n_test=250, rollout_every=200, topk k=3 by SR.
#
# Usage (cluster docker):
#   bash scripts/cluster_launch_metaworld_single_policy_4.sh
set -euo pipefail
cd /workspace/oat

declare -A TOK_CKPT=(
  [box-close]="output/20260710/212943_train_oattok_mw-box-close_st_N50/checkpoints/ep-3450_mse-0.019.ckpt"
  [coffee-pull]="output/20260710/212943_train_oattok_mw-coffee-pull_st_N50/checkpoints/ep-2670_mse-0.039.ckpt"
  [disassemble]="output/20260710/235437_train_oattok_mw-disassemble_st_N50/checkpoints/ep-3410_mse-0.027.ckpt"
  [stick-pull]="output/20260710/235437_train_oattok_mw-stick-pull_st_N50/checkpoints/ep-3030_mse-0.042.ckpt"
)

TASKS=(box-close coffee-pull disassemble stick-pull)
GPUS=(0 1 0 1)
SEED="${SEED:-0}"
NUM_DEMO="${NUM_DEMO:-50}"
ROLLOUT_EVERY="${ROLLOUT_EVERY:-200}"
N_TEST="${N_TEST:-250}"
N_PARALLEL_ENVS="${N_PARALLEL_ENVS:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"

echo "=== Launch 4 single-task MetaWorld policy runs (best tokenizer ckpts) ==="
echo "rollout_every=${ROLLOUT_EVERY} n_test=${N_TEST} topk=3 (full paper eval, no fast mode)"

for i in "${!TASKS[@]}"; do
  task="${TASKS[$i]}"
  gpu="${GPUS[$i]}"
  tok="${TOK_CKPT[$task]}"
  sess="mwst_pol_${task//-/_}"

  if [[ ! -f "${tok}" ]]; then
    echo "[ERROR] missing tokenizer ckpt for ${task}: ${tok}"
    exit 1
  fi

  if tmux has-session -t "${sess}" 2>/dev/null; then
    echo "[SKIP] session exists: ${sess}"
    continue
  fi

  cmd="TASK=${task} GPU=${gpu} SEED=${SEED} NUM_DEMO=${NUM_DEMO} \
TOKENIZER_CKPT=${tok} \
ROLLOUT_EVERY=${ROLLOUT_EVERY} N_TEST=${N_TEST} \
N_PARALLEL_ENVS=${N_PARALLEL_ENVS} NUM_WORKERS=${NUM_WORKERS} \
bash scripts/cluster_policy_metaworld_single.sh"

  tmux new-session -d -s "${sess}" "${cmd}"
  echo "[OK] ${sess} GPU${gpu} task=${task} tok=$(basename "${tok}")"
done

echo ""
tmux ls 2>/dev/null || true
