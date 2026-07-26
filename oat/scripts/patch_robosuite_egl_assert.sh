#!/usr/bin/env bash
# Allow MUJOCO_EGL_DEVICE_ID=0 when CUDA_VISIBLE_DEVICES exposes a single GPU
# (e.g. CUDA_VISIBLE_DEVICES=1 → cuda:0 is physical GPU1, EGL has one device at index 0).
# Robosuite's stock check requires MUJOCO_EGL_DEVICE_ID to appear in CUDA_VISIBLE_DEVICES,
# which breaks single-GPU isolation on non-zero GPUs.
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV:-.venv}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

TARGET="$(python - <<'PY'
import robosuite.utils.binding_utils as m
print(m.__file__)
PY
)"

OLD='        assert MUJOCO_EGL_DEVICE_ID.isdigit() and (
            MUJOCO_EGL_DEVICE_ID in CUDA_VISIBLE_DEVICES
        ), "MUJOCO_EGL_DEVICE_ID needs to be set to one of the device id specified in CUDA_VISIBLE_DEVICES"'

NEW='        _cvd_parts = [p.strip() for p in CUDA_VISIBLE_DEVICES.split(",") if p.strip()]
        assert MUJOCO_EGL_DEVICE_ID.isdigit() and (
            MUJOCO_EGL_DEVICE_ID in CUDA_VISIBLE_DEVICES
            or (MUJOCO_EGL_DEVICE_ID == "0" and len(_cvd_parts) == 1)
        ), "MUJOCO_EGL_DEVICE_ID needs to be set to one of the device id specified in CUDA_VISIBLE_DEVICES"'

if grep -q '_cvd_parts' "${TARGET}"; then
  echo "robosuite EGL assert already patched: ${TARGET}"
  exit 0
fi

python - <<PY
from pathlib import Path
path = Path("${TARGET}")
text = path.read_text()
old = '''        assert MUJOCO_EGL_DEVICE_ID.isdigit() and (
            MUJOCO_EGL_DEVICE_ID in CUDA_VISIBLE_DEVICES
        ), "MUJOCO_EGL_DEVICE_ID needs to be set to one of the device id specified in CUDA_VISIBLE_DEVICES"'''
new = '''        _cvd_parts = [p.strip() for p in CUDA_VISIBLE_DEVICES.split(",") if p.strip()]
        assert MUJOCO_EGL_DEVICE_ID.isdigit() and (
            MUJOCO_EGL_DEVICE_ID in CUDA_VISIBLE_DEVICES
            or (MUJOCO_EGL_DEVICE_ID == "0" and len(_cvd_parts) == 1)
        ), "MUJOCO_EGL_DEVICE_ID needs to be set to one of the device id specified in CUDA_VISIBLE_DEVICES"'''
if old not in text:
    raise SystemExit(f"pattern not found in {path}")
path.write_text(text.replace(old, new, 1))
print(f"patched {path}")
PY
