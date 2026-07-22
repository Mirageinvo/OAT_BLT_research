#!/usr/bin/env bash
# 03 — convert demos -> Zarr. REQUIRES you filled the converter TODOs (from 02 output).
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
echo "[03] converting ${SRC_DIR} (${DATA_FORMAT}) -> ${ZARR}"
uv run python scripts/convert_robotwin_dataset.py \
  --src "${SRC_DIR}" --task "${TASK}" -n "${NDEMO}" --format "${DATA_FORMAT}"
echo "[03] done. NEXT: 04_verify."
