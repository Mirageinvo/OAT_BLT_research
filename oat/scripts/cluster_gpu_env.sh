#!/usr/bin/env bash
# Source from cluster scripts:  source scripts/cluster_gpu_env.sh <physical_gpu_id>
# physical_gpu_id: 0 or 1 on the 2×V100 node.
#
# GPU0 jobs (square train, can eval): CUDA_VISIBLE_DEVICES=0, EGL=0, cuda:0
# GPU1 jobs (mt4 train, lift eval):   CUDA_VISIBLE_DEVICES=1, EGL=0 (after robosuite patch), cuda:0
#
# For lift eval we also support the explicit two-GPU visibility path (no patch needed):
#   source scripts/cluster_gpu_env.sh 1 two_gpu_visible
set -euo pipefail

PHYSICAL_GPU="${1:?usage: source cluster_gpu_env.sh <0|1> [two_gpu_visible]}"
MODE="${2:-isolated}"

export MUJOCO_GL=egl
export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"

if [[ "${PHYSICAL_GPU}" == "0" ]]; then
  export CUDA_VISIBLE_DEVICES=0
  export MUJOCO_EGL_DEVICE_ID=0
  export OAT_DEVICE=cuda:0
elif [[ "${PHYSICAL_GPU}" == "1" && "${MODE}" == "two_gpu_visible" ]]; then
  export CUDA_VISIBLE_DEVICES=0,1
  export MUJOCO_EGL_DEVICE_ID=1
  export OAT_DEVICE=cuda:1
elif [[ "${PHYSICAL_GPU}" == "1" ]]; then
  # Single-GPU isolation on physical GPU1 (needs patch_robosuite_egl_assert.sh once).
  export CUDA_VISIBLE_DEVICES=1
  export MUJOCO_EGL_DEVICE_ID=0
  export OAT_DEVICE=cuda:0
else
  echo "ERROR: PHYSICAL_GPU must be 0 or 1, got ${PHYSICAL_GPU}" >&2
  return 1 2>/dev/null || exit 1
fi
