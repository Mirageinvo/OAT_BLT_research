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
ls script/ || true
# RoboTwin install (verify names in script/): install + assets download (~20 min)
bash script/_install.sh          || echo "!! _install.sh failed -> manual fallback (README): pip install -r requirements.txt; pytorch3d; curobo; mplib fix"
bash script/_download_assets.sh  || echo "!! _download_assets.sh failed -> download assets manually"
# RoboTwin task modules must be importable:
export PYTHONPATH="${PYTHONPATH:-}:${ROBOTWIN_DIR}"
echo "export PYTHONPATH=\"\${PYTHONPATH}:${ROBOTWIN_DIR}\"  # add to ~/.bashrc"
cd "${OAT_DIR}"
echo "[00] done. SANITY: run a RoboTwin demo/eval from ${ROBOTWIN_DIR} to confirm SAPIEN renders."
