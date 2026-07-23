#!/usr/bin/env bash
# Isolated venv for demo_v15 -> image extract (robosuite 1.5 + robomimic >=0.4).
# Does NOT modify the train .venv (robosuite 1.4 + mujoco210 for Lift policy).
#
# Usage:
#   cd /workspace/oat
#   bash scripts/setup_extract_venv.sh
#   FORCE=1 bash scripts/setup_extract_venv.sh   # wipe + recreate

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${ROOT}/.venv_extract"
FORCE="${FORCE:-0}"

venv_ok() {
  [[ -x "${VENV}/bin/python" ]] || return 1
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
  # robosuite 1.5.1 needs mujoco>=3.2.3; mujoco>=3.10 breaks mj_fullM (OSC)
  python -c "
import robosuite, robomimic, mujoco, torchvision, diffusers
v = tuple(map(int, mujoco.__version__.split('.')[:2]))
assert v >= (3, 2) and v < (3, 10), mujoco.__version__
" 2>/dev/null
}

if [[ "${FORCE}" == "1" ]]; then
  echo "FORCE=1: removing ${VENV}"
  rm -rf "${VENV}"
fi

if venv_ok; then
  echo "exists + healthy: ${VENV}"
  python -c "import robosuite, robomimic, mujoco; print('robosuite', robosuite.__version__, 'robomimic', robomimic.__version__, 'mujoco', mujoco.__version__)"
  exit 0
fi

if [[ -d "${VENV}" ]]; then
  echo "broken/incomplete ${VENV} — recreating"
  rm -rf "${VENV}"
fi

python3 -m venv "${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
pip install -U pip wheel
# robosuite 1.5.1: mujoco>=3.2.3, but <3.10 (3.10 changed mj_fullM → OSC TypeError)
pip install "robosuite==1.5.1" "mujoco>=3.2.3,<3.10" h5py numpy tqdm imageio egl_probe opencv-python
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install "git+https://github.com/ARISE-Initiative/robomimic.git@master" --no-deps
# robomimic algo/__init__ eagerly imports diffusion_policy → needs these even for extract
pip install "diffusers==0.11.1" tensorboard
pip install transformers==4.41.2 huggingface_hub==0.23.4 imageio-ffmpeg matplotlib psutil termcolor tensorboardX

python -c "import robosuite, robomimic, mujoco; print('robosuite', robosuite.__version__, 'robomimic', robomimic.__version__, 'mujoco', mujoco.__version__)"
