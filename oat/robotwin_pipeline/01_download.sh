#!/usr/bin/env bash
# 01 — download demos for ${TASK}. Two options via DATA_FORMAT.
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
mkdir -p "${SRC_DIR}"

if [ "${DATA_FORMAT}" = "lerobot" ]; then
  echo "[01] LeRobot download (HF): ${HF_REPO} -> ${SRC_DIR}"
  huggingface-cli download --repo-type dataset "${HF_REPO}" --local-dir "${SRC_DIR}"
else
  echo "[01] native HDF5 collection for ${TASK} (${NDEMO} eps)"
  cd "${ROBOTWIN_DIR}"
  # VERIFY args against robotwin-platform.github.io/doc/usage/collect-data.html
  python script/collect_data.py --task "${TASK}" --num_episodes "${NDEMO}" || \
    echo "!! adjust collect_data args per RoboTwin docs; output HDF5s should go under ${SRC_DIR}"
  cd "${OAT_DIR}"
fi
echo "[01] done -> ${SRC_DIR}. NEXT: run 02_inspect."
