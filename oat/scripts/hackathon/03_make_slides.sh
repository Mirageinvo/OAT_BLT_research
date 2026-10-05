#!/usr/bin/env bash
# Verify HF baseline assets are present and print the paper table.
set -euo pipefail
cd "$(dirname "$0")/../.."

STAGE="${STAGE:-hackathon_assets}"
OUT="${OUT:-hackathon_output}"

python3 scripts/hackathon/validate_baseline_assets.py \
  --stage "${STAGE}" --out "${OUT}"

python3 scripts/hackathon/make_baseline_slide.py \
  --out_dir "${OUT}/progress" --eval_root "${OUT}"

echo "Slide cards -> ${OUT}/progress/baseline_training_cards.png"
