#!/usr/bin/env bash
# 00 — install OAT deps + RoboTwin/SAPIEN (isolated env recommended). Run once.
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

echo "[00] OAT deps"
uv sync

echo "[00] RoboTwin at ${ROBOTWIN_DIR}"
if [ ! -d "${ROBOTWIN_DIR}" ]; then
  git clone https://github.com/robotwin-Platform/robotwin "${ROBOTWIN_DIR}"
fi
cd "${ROBOTWIN_DIR}"
# VERIFY the actual install entrypoint in RoboTwin's README (script name may differ):
bash script/install.sh || echo "!! run RoboTwin install manually per their README, then re-run 01"
cd "${OAT_DIR}"
echo "[00] done. SANITY: run a RoboTwin demo/eval from ${ROBOTWIN_DIR} to confirm SAPIEN renders."
