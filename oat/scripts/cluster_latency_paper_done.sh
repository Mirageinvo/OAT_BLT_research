#!/usr/bin/env bash
# Paper Table C latency for Wave2-done suites (RESOLUTIONPLAN § Latency).
# Paper-proof: requires OAT_GIT_* (docker has no .git). Does NOT touch Square collect.
#
# FAIR_KV=1 → apples-to-apples KV-cache Single/AWR (writes latency_fair_kv.json;
#             does NOT overwrite Table C latency.json). Default = deployed Table C.
#
# Usage:
#   SUITES="can coffee-pull ..." GPU=0 bash scripts/cluster_latency_paper_done.sh
#   FAIR_KV=1 SUITES="can coffee-pull stick-pull disassemble box-close square" GPU=0 \
#     bash scripts/cluster_latency_paper_done.sh
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
FAIR_KV="${FAIR_KV:-0}"
if [[ "${FAIR_KV}" == "1" ]]; then
  LOG="logs/latency_fair_kv_s10000_gpu${GPU}.log"
  EXTRA_FLAGS=(--fair_kv)
  LABEL="fair-KV"
else
  LOG="logs/latency_paper_s10000_gpu${GPU}.log"
  EXTRA_FLAGS=()
  LABEL="paper-proof deployed"
fi
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
  echo "=== PAPER LATENCY ${LABEL} $(date -Iseconds) gpu=${CUDA_VISIBLE_DEVICES} device=${OAT_DEVICE} ==="
  echo "suites=${SUITES} reps=${REPS} warmup=${WARMUP} FAIR_KV=${FAIR_KV}"
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
      "${EXTRA_FLAGS[@]}" \
      2>&1 | tee -a "${LOG}"
done

echo "=== LATENCY ALL DONE $(date -Iseconds) ===" | tee -a "${LOG}"
if [[ "${FAIR_KV}" == "1" ]]; then
  python scripts/build_table_c.py --fair_kv 2>&1 | tee -a "${LOG}"
else
  python scripts/build_table_c.py 2>&1 | tee -a "${LOG}"
fi
