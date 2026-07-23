#!/usr/bin/env bash
# Can: rebuild extract venv → ensure raw demo_v15 → image HDF5 → replan probe.
# Usage (cluster docker): bash scripts/can_extract_pipeline.sh
set -euo pipefail

cd "$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p logs data/robomimic/hdf5_datasets/can/mh
LOG=logs/can_extract_pipeline.log
trap 'ec=$?; echo "EXIT=$ec" | tee -a "$LOG"; exit "$ec"' ERR

log() { echo "$*" | tee -a "$LOG"; }

log "=== CAN EXTRACT PIPELINE $(date -Iseconds) ==="

log "[1] ensure .venv_extract (mujoco>=3.2.3,<3.10 + torchvision)"
# Recreate only if unhealthy / wrong mujoco; FORCE=1 to wipe explicitly.
FORCE="${FORCE:-0}" bash scripts/setup_extract_venv.sh 2>&1 | tee -a "$LOG"
# shellcheck disable=SC1091
source .venv_extract/bin/activate
# torchvision / diffusers required by robomimic import graph (even for extract-only)
python -c "import torchvision" 2>/dev/null || \
  pip install torchvision --index-url https://download.pytorch.org/whl/cpu 2>&1 | tee -a "$LOG"
python -c "import diffusers" 2>/dev/null || \
  pip install "diffusers==0.11.1" tensorboard 2>&1 | tee -a "$LOG"
python -c "
import mujoco, robosuite, torchvision, diffusers
v = tuple(map(int, mujoco.__version__.split('.')[:2]))
assert v >= (3, 2) and v < (3, 10), mujoco.__version__
print('OK', mujoco.__version__, robosuite.__version__,
      'torchvision', torchvision.__version__, 'diffusers', diffusers.__version__)
" 2>&1 | tee -a "$LOG"

RAW=data/robomimic/hdf5_datasets/can/mh/demo_v15.hdf5
if [[ -f "$RAW" ]]; then
  log "[2] raw exists ($(du -h "$RAW" | cut -f1))"
else
  log "[2] download raw"
  curl -L --fail --retry 3 -o "$RAW" \
    https://huggingface.co/datasets/robomimic/robomimic_datasets/resolve/main/v1.5/can/mh/demo_v15.hdf5 \
    2>&1 | tee -a "$LOG"
fi

log "[3] extract"
export MUJOCO_GL=egl OAT_USE_UV_RUN=0
bash scripts/extract_robomimic_mh_image.sh can 2>&1 | tee -a "$LOG"

IMG=data/robomimic/hdf5_datasets/can_mh_image.hdf5
[[ -f "$IMG" ]] || { log "FAIL: missing $IMG"; exit 1; }
log "[3] OK $(du -h "$IMG" | cut -f1)"

log "[4] replan Can"
CKPT="$(find output -path '*can*N200*/checkpoints/ep-*_sr-*.ckpt' 2>/dev/null | sort | tail -1 || true)"
[[ -n "$CKPT" ]] || { log "FAIL: no can N200 ckpt"; exit 1; }
log "ckpt=$CKPT"
# Do NOT mkdir OUT — eval_policy_sim prompts Overwrite if the dir already exists.
rm -rf eval_out/replan_probe_can
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl .venv/bin/python scripts/eval_policy_sim.py \
  -c "$CKPT" -o eval_out/replan_probe_can -n 1 \
  --use_k_tokens 8 --entropy_threshold 0 \
  --n_test 10 --n_parallel_envs 1 \
  2>&1 | tee -a logs/replan_probe_can.log | tee -a "$LOG"

log "=== DONE $(date -Iseconds) ==="
echo "EXIT=0" | tee -a "$LOG"
