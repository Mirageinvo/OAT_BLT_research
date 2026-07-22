#!/usr/bin/env bash
# Isolated RoboCasa venv (robosuite 1.5.1 + robocasa v0.2).
# NEVER installs into /workspace/oat/.venv (shared Lift/Square/Can/MW = robosuite 1.4).
#
# Usage (inside oat docker):
#   bash scripts/cluster_setup_robocasa_venv.sh
#
# Pins: torch==2.5.1+cu124 (V100), numpy==1.23.5 (robocasa assert), robosuite 1.5.1.
# Skips metaworld/libero. Installs robocasa editable from git so scripts/ ship.
set -euo pipefail
cd /workspace/oat
mkdir -p logs
LOG="${LOG:-logs/cluster_setup_robocasa_venv.log}"
VENV="${VENV:-/workspace/oat/.venv_robocasa}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

exec > >(tee -a "${LOG}") 2>&1
echo "=== robocasa venv setup | $(date -Iseconds) ==="
echo "venv=${VENV}"
echo "NOTE: will NOT touch /workspace/oat/.venv"

if [[ -d /workspace/oat/.venv ]] && [[ "${VENV}" -ef /workspace/oat/.venv ]]; then
  echo "ERROR: refusing to use shared .venv"
  exit 1
fi

df -h /workspace / 2>/dev/null || true

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "[1/5] create ${VENV}"
  uv venv "${VENV}"
else
  echo "[1/5] reuse existing ${VENV}"
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
export OAT_USE_UV_RUN=0
PY="${VENV}/bin/python"

uv_pip() {
  UV_PROJECT_ENVIRONMENT="${VENV}" uv pip "$@"
}

echo "[2/5] oat editable --no-deps"
uv_pip install -e . --no-deps

echo "[2a/5] torch 2.5.1+cu124 (match shared .venv / V100)"
uv_pip install \
  "torch==2.5.1" "torchvision==0.20.1" \
  --index-url https://download.pytorch.org/whl/cu124

echo "[2b/5] core deps (no metaworld/libero/robosuite)"
uv_pip install \
  "einops>=0.8.1" \
  "hydra-core>=1.3.2" \
  "wandb>=0.24.0" \
  "dill>=0.4.1" \
  "zarr>=2.18.3" \
  "diffusers>=0.36.0" \
  "accelerate>=1.12.0" \
  "transformers>=4.57.6" \
  "av>=16.1.0" \
  "gymnasium>=1.2.3" \
  "vector-quantize-pytorch>=1.27.19" \
  "robomimic==0.3.0" \
  "easydict>=1.13" \
  "bddl>=3.6.0" \
  "cloudpickle>=3.1.2" \
  "matplotlib>=3.10.8" \
  "gym>=0.26.2" \
  "pandas>=2.3.3" \
  "h5py" \
  "tqdm" \
  "termcolor" \
  "imageio" \
  "imageio-ffmpeg" \
  "scipy" \
  "opencv-python-headless"

echo "[3/5] robosuite==1.5.1 + robocasa@v0.2 (editable, includes scripts/)"
uv_pip install --upgrade "robosuite==1.5.1"
# Wheel omits scripts/; install from git checkout / clone so create_env + asset download work.
ROBOCASA_SRC="${ROBOCASA_SRC:-}"
if [[ -z "${ROBOCASA_SRC}" ]]; then
  uv_pip install --upgrade "robocasa @ git+https://github.com/robocasa/robocasa.git@v0.2"
  # Prefer editable reinstall from uv git checkout if present.
  CAND="$(find /home/askhabaliev_gs/.cache/uv/git-v0/checkouts -path '*/robocasa/scripts/download_kitchen_assets.py' 2>/dev/null | head -1 || true)"
  if [[ -n "${CAND}" ]]; then
    ROBOCASA_SRC="$(cd "$(dirname "${CAND}")/../.." && pwd)"
  fi
fi
if [[ -n "${ROBOCASA_SRC}" && -d "${ROBOCASA_SRC}" ]]; then
  echo "editable robocasa from ${ROBOCASA_SRC}"
  uv_pip install --force-reinstall --no-deps -e "${ROBOCASA_SRC}"
fi

echo "[3b/5] re-pin torch + numpy==1.23.5 (robocasa hard assert) + matching numba"
uv_pip install --force-reinstall \
  "torch==2.5.1" "torchvision==0.20.1" \
  --index-url https://download.pytorch.org/whl/cu124
uv_pip install --force-reinstall \
  "numpy==1.23.5" "numba==0.56.4" "llvmlite==0.39.1"

echo "[4/5] verify imports"
"${PY}" - <<'PY'
import numpy, torch, robosuite, robocasa
print("numpy", numpy.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("robosuite", robosuite.__version__)
print("robocasa", getattr(robocasa, "__version__", "?"), "path=", robocasa.__file__)
assert robosuite.__version__.startswith("1.5"), robosuite.__version__
assert torch.__version__.startswith("2.5"), torch.__version__
assert numpy.__version__ in ("1.23.2", "1.23.3", "1.23.5"), numpy.__version__
from robocasa.utils.env_utils import create_env
print("create_env OK")
import oat
print("oat OK", oat.__file__)
PY

echo "[5/5] kitchen assets"
"${PY}" - <<'PY'
import importlib.util
import pathlib
import subprocess
import sys

rc = pathlib.Path(importlib.util.find_spec("robocasa").origin).resolve().parent
scripts = rc / "scripts"
macros = scripts / "setup_macros.py"
assets = scripts / "download_kitchen_assets.py"
if macros.is_file():
    print("running", macros)
    subprocess.check_call([sys.executable, str(macros)])
if assets.is_file():
    print("running", assets, "(pipes y to interactive prompt)")
    # Official script has no --yes; it always prompts.
    subprocess.run(
        [sys.executable, str(assets)],
        input="y\n",
        text=True,
        check=True,
    )
else:
    raise SystemExit(f"download_kitchen_assets.py missing under {scripts}")
PY

echo "=== DONE $(date -Iseconds) | source ${VENV}/bin/activate ==="
df -h /workspace / 2>/dev/null || true
