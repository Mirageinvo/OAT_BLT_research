#!/usr/bin/env bash
# Isolated venv for demo_v15 -> image extract (robosuite 1.5 + robomimic >=0.4).
# Does NOT modify the train .venv (robosuite 1.4 + mujoco210 for Lift policy).
#
# Usage:
#   cd /workspace/oat
#   bash scripts/setup_extract_venv.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${ROOT}/.venv_extract"

if [[ -d "${VENV}" ]]; then
  echo "exists: ${VENV}"
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
  python -c "import robosuite, robomimic; print('robosuite', robosuite.__version__, 'robomimic', robomimic.__version__)"
  exit 0
fi

python3 -m venv "${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
pip install -U pip wheel
pip install "robosuite==1.5.1" "mujoco==3.1.6" h5py numpy tqdm imageio egl_probe opencv-python
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "git+https://github.com/ARISE-Initiative/robomimic.git@master" --no-deps
pip install transformers==4.41.2 huggingface_hub==0.23.4 imageio-ffmpeg matplotlib psutil termcolor tensorboardX

python -c "import robosuite, robomimic, mujoco; print('robosuite', robosuite.__version__, 'robomimic', robomimic.__version__, 'mujoco', mujoco.__version__)"
