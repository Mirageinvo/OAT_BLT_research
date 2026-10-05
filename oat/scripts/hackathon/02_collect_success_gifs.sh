#!/usr/bin/env bash
# Collect ≥1 success gif per task from freshly trained policies (no SR reporting).
set -euo pipefail
cd "$(dirname "$0")/../.."

export MUJOCO_GL="${MUJOCO_GL:-egl}"
POL_DIR="${POL_DIR:-hackathon_assets/policies}"
GIF_DIR="${GIF_DIR:-hackathon_output/gifs}"
TARGET="${TARGET:-1}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-80}"

pick_ckpt() {
  local task="$1"
  local d="${POL_DIR}/${task}"
  if [[ ! -d "${d}" ]]; then
    echo "Missing ${d}. Run 01_train_policies.sh first."
    exit 1
  fi
  ls -t "${d}"/ep-*_sr-*.ckpt 2>/dev/null | head -1 || ls -t "${d}"/*.ckpt | head -1
}

mkdir -p "${GIF_DIR}"

for task in lift can square; do
  ckpt="$(pick_ckpt "${task}")"
  echo "=== ${task}: ${ckpt} ==="
  uv run python scripts/hackathon/rollout_until_success.py \
    -c "${ckpt}" \
    -o "${GIF_DIR}" \
    --task "${task}" \
    --target "${TARGET}" \
    --max_attempts "${MAX_ATTEMPTS}"
done

echo "Clips ready under ${GIF_DIR}/"
