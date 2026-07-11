#!/usr/bin/env bash
set -euo pipefail

cd /workspace/oat
source .venv/bin/activate

export OAT_USE_UV_RUN=0
export MUJOCO_GL=egl
# shellcheck disable=SC1091
source scripts/cluster_gpu_env.sh 0

LOG="logs/metaworld_single_data_regen.log"
mkdir -p logs
echo "[START] $(date -Iseconds)" | tee "${LOG}"

for t in box-close coffee-pull disassemble stick-pull; do
  echo "[GEN] task=${t}" | tee -a "${LOG}"
  python scripts/gen_metaworld_data.py \
    --task_name "${t}" \
    --num_episodes 50 \
    --device "${OAT_DEVICE}" \
    --force 2>&1 | tee -a "${LOG}"

  python scripts/validate_metaworld_data.py \
    "data/metaworld/${t}_N50.zarr" \
    --episodes-per-task 50 \
    --num-tasks 1 2>&1 | tee -a "${LOG}"
done

echo "[DONE] $(date -Iseconds)" | tee -a "${LOG}"
