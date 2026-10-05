#!/usr/bin/env bash
# Re-plot training progress from existing run dirs (or after 01_train_policies.sh).
set -euo pipefail
cd "$(dirname "$0")/../.."

PROG="${PROG:-hackathon_output/progress}"
mkdir -p "${PROG}"

plot_task() {
  local task="$1"
  local run_dir
  run_dir="$(ls -td output/*/*train_oatpolicy*${task}* 2>/dev/null | head -1 || true)"
  if [[ -z "${run_dir}" || ! -f "${run_dir}/logs.json" ]]; then
    echo "WARN: no logs.json for ${task} (skip)"
    return 0
  fi
  uv run python scripts/hackathon/plot_training_progress.py \
    --run_dir "${run_dir}" \
    --task "${task}" \
    --output_dir "${PROG}"
}

for task in lift can square; do
  plot_task "${task}"
done

# combined slide (3 panels)
uv run python scripts/hackathon/plot_training_progress_combined.py \
  --progress_dir "${PROG}" \
  -o "${PROG}/all_tasks_training.png" 2>/dev/null || true

echo "Progress plots -> ${PROG}/"
