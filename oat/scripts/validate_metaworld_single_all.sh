#!/usr/bin/env bash
set -euo pipefail

cd /workspace/oat
source .venv/bin/activate

TASKS=(box-close coffee-pull disassemble stick-pull)
BASE_PATH="data/metaworld"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-50}"

echo "=== Validating all single-task MetaWorld datasets ==="

for TASK in "${TASKS[@]}"; do
  ZARR_PATH="${BASE_PATH}/${TASK}_N${EPISODES_PER_TASK}.zarr"
  echo "-> Checking ${ZARR_PATH} ..."

  if [[ ! -d "${ZARR_PATH}" ]]; then
    echo "ERROR: missing dataset ${ZARR_PATH}"
    exit 1
  fi

  python scripts/validate_metaworld_data.py \
    "${ZARR_PATH}" \
    --episodes-per-task "${EPISODES_PER_TASK}" \
    --num-tasks 1

  echo "OK: ${TASK}"
  echo ""
done

echo "=== All 4 tasks passed validation ==="
