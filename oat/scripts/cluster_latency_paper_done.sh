#!/usr/bin/env bash
# Paper Table C latency for Wave2-done suites (RESOLUTIONPLAN § Latency).
# Paper-proof: requires OAT_GIT_* (docker has no .git). Does NOT touch Square collect.
# Uses GPU0 by default (Square often on GPU1).
set -euo pipefail
cd /workspace/oat
source .venv/bin/activate
export MUJOCO_GL=egl
export OAT_USE_UV_RUN=0

GPU="${GPU:-0}"
# shellcheck disable=SC1091
source scripts/cluster_gpu_env.sh "${GPU}"
bash scripts/patch_robosuite_egl_assert.sh

SUITES="${SUITES:-can coffee-pull stick-pull disassemble box-close}"
REPS="${REPS:-10}"
WARMUP="${WARMUP:-20}"
LOG="logs/latency_paper_s10000_gpu${GPU}.log"
mkdir -p logs

if [[ -z "${OAT_GIT_COMMIT:-}" ]]; then
  echo "ERROR: OAT_GIT_COMMIT unset — refuse paper-proof latency without git provenance." >&2
  echo "  Launch from host: export OAT_GIT_COMMIT=\$(git -C ~/mipt_paper rev-parse HEAD)" >&2
  exit 1
fi
export OAT_GIT_COMMIT
export OAT_GIT_BRANCH="${OAT_GIT_BRANCH:-unknown}"
export OAT_GIT_DIRTY="${OAT_GIT_DIRTY:-0}"

{
  echo "=== PAPER LATENCY paper-proof $(date -Iseconds) gpu=${CUDA_VISIBLE_DEVICES} device=${OAT_DEVICE} ==="
  echo "suites=${SUITES} reps=${REPS} warmup=${WARMUP}"
  echo "OAT_GIT_COMMIT=${OAT_GIT_COMMIT} branch=${OAT_GIT_BRANCH} dirty=${OAT_GIT_DIRTY}"
} | tee "${LOG}"

for s in ${SUITES}; do
  echo "" | tee -a "${LOG}"
  echo "[LATENCY] ${s} $(date -Iseconds)" | tee -a "${LOG}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl \
    OAT_GIT_COMMIT="${OAT_GIT_COMMIT}" OAT_GIT_BRANCH="${OAT_GIT_BRANCH}" OAT_GIT_DIRTY="${OAT_GIT_DIRTY}" \
    python scripts/measure_latency_paper.py \
      --suite "${s}" \
      -d "${OAT_DEVICE}" \
      --reps "${REPS}" \
      --warmup "${WARMUP}" \
      2>&1 | tee -a "${LOG}"
done

echo "=== LATENCY ALL DONE $(date -Iseconds) ===" | tee -a "${LOG}"
python scripts/build_table_c.py 2>&1 | tee -a "${LOG}"
