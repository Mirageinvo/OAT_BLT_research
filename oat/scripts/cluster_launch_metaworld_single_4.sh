#!/usr/bin/env bash
# Launch 4 parallel single-task MetaWorld specialist pipelines in tmux.
# Does not stop existing sessions; skips tasks whose session name already exists.
set -euo pipefail
cd /workspace/oat

TASKS=(box-close coffee-pull disassemble stick-pull)
GPUS=(0 1 0 1)
SEED="${SEED:-0}"
NUM_DEMO="${NUM_DEMO:-50}"

echo "=== Launching 4 single-task MetaWorld pipelines ==="
for i in "${!TASKS[@]}"; do
  task="${TASKS[$i]}"
  gpu="${GPUS[$i]}"
  sess="mwst_${task//-/_}"

  if tmux has-session -t "${sess}" 2>/dev/null; then
    echo "[SKIP] session exists: ${sess}"
    continue
  fi

  cmd="TASK=${task} GPU=${gpu} SEED=${SEED} NUM_DEMO=${NUM_DEMO} bash scripts/cluster_metaworld_single_full_pipeline.sh"
  tmux new-session -d -s "${sess}" "${cmd}"
  echo "[OK] started ${sess} on GPU${gpu} (task=${task})"
done

echo ""
echo "tmux sessions:"
tmux ls
