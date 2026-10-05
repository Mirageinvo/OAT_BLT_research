#!/usr/bin/env bash
# Table C′ extension: Single (opt) + BoN{8,16,32}, NO AWR.
# Writes latency_fair_kv_n16.json — does NOT overwrite locked latency_fair_kv.json.
#
# Usage (inside docker /workspace/oat):
#   GPU=0 BON_NS=16,32 INCLUDE_SINGLE=0 VENV=.venv \
#     SUITES="can lift square coffee-pull stick-pull disassemble box-close" \
#     bash scripts/cluster_latency_cprime_no_awr.sh
#   GPU=0 VENV=.venv_robocasa BON_NS="" INCLUDE_SINGLE=1 \
#     SUITES="coffee_press_button close_drawer turn_off_sink_faucet turn_off_microwave" \
#     bash scripts/cluster_latency_cprime_no_awr.sh
set -euo pipefail
cd /workspace/oat

VENV="${VENV:-.venv}"
# Docker often has .venv/bin/python → host uv path (ENOENT). Fall back to bundled cpython + site-packages.
RUNTIME_PY="${RUNTIME_PY:-/workspace/oat/_runtime_uv_python/cpython-3.10.20-linux-x86_64-gnu/bin/python3.10}"
if [[ -x "${VENV}/bin/python" ]] && "${VENV}/bin/python" -c "import hydra" 2>/dev/null; then
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
  PYTHON="${VENV}/bin/python"
else
  echo "[latency] ${VENV}/bin/python broken or missing hydra — using RUNTIME_PY + ${VENV} site-packages"
  export VIRTUAL_ENV
  VIRTUAL_ENV="$(cd "${VENV}" && pwd)"
  export VIRTUAL_ENV
  export PATH="${VIRTUAL_ENV}/bin:${PATH}"
  export PYTHONPATH="/workspace/oat/_runtime_venv_boot:${VIRTUAL_ENV}/lib/python3.10/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
  PYTHON="${RUNTIME_PY}"
fi
export MUJOCO_GL=egl
export OAT_USE_UV_RUN=0

GPU="${GPU:-0}"
# shellcheck disable=SC1091
source scripts/cluster_gpu_env.sh "${GPU}"
bash scripts/patch_robosuite_egl_assert.sh || true

SUITES="${SUITES:-can lift square coffee-pull stick-pull disassemble box-close}"
REPS="${REPS:-10}"
TRIALS="${TRIALS:-8}"
WARMUP="${WARMUP:-20}"
BON_NS="${BON_NS-16,32}"  # allow empty BON_NS="" for Single-only
INCLUDE_SINGLE="${INCLUDE_SINGLE:-0}"
LOG="logs/latency_fair_kv_n16_gpu${GPU}.log"
mkdir -p logs

if [[ -z "${OAT_GIT_COMMIT:-}" ]]; then
  echo "ERROR: OAT_GIT_COMMIT unset" >&2
  exit 1
fi
export OAT_GIT_COMMIT OAT_GIT_BRANCH="${OAT_GIT_BRANCH:-unknown}" OAT_GIT_DIRTY="${OAT_GIT_DIRTY:-0}"

SINGLE_FLAG=(--no_single)
if [[ "${INCLUDE_SINGLE}" == "1" ]]; then
  SINGLE_FLAG=(--include_single)
fi

BON_FLAG=(--bon_ns "${BON_NS}")
if [[ -z "${BON_NS}" ]]; then
  BON_FLAG=(--bon_ns "")
fi

declare -A RC_CKPT=(
  [coffee_press_button]="my_models/robocasa_coffee_press_button_topk_ep0500_sr0.600.ckpt"
  [close_drawer]="my_models/robocasa_close_drawer_topk_ep0500_sr0.700.ckpt"
  [turn_off_sink_faucet]="my_models/robocasa_turn_off_sink_faucet_topk_ep0500_sr0.580.ckpt"
  [turn_off_microwave]="my_models/robocasa_turn_off_microwave_topk_ep0500_sr0.620.ckpt"
)

{
  echo "=== C′ no-AWR LATENCY $(date -Iseconds) gpu=${CUDA_VISIBLE_DEVICES} device=${OAT_DEVICE} venv=${VENV} python=${PYTHON} ==="
  echo "suites=${SUITES} bon_ns=${BON_NS} include_single=${INCLUDE_SINGLE} trials=${TRIALS} reps=${REPS}"
  echo "OAT_GIT_COMMIT=${OAT_GIT_COMMIT}"
} | tee "${LOG}"

for s in ${SUITES}; do
  echo "" | tee -a "${LOG}"
  echo "[LATENCY] ${s} $(date -Iseconds)" | tee -a "${LOG}"
  EXTRA=()
  if [[ -n "${RC_CKPT[$s]:-}" ]]; then
    EXTRA+=(--base_ckpt "${RC_CKPT[$s]}")
  fi
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl \
    OAT_GIT_COMMIT="${OAT_GIT_COMMIT}" OAT_GIT_BRANCH="${OAT_GIT_BRANCH}" OAT_GIT_DIRTY="${OAT_GIT_DIRTY}" \
    PYTHONPATH="${PYTHONPATH:-}" VIRTUAL_ENV="${VIRTUAL_ENV:-}" \
    "${PYTHON}" scripts/measure_latency_paper.py \
      --suite "${s}" \
      -d "${OAT_DEVICE}" \
      --reps "${REPS}" \
      --trials "${TRIALS}" \
      --warmup "${WARMUP}" \
      --fair_kv \
      --skip_awr \
      "${BON_FLAG[@]}" \
      "${SINGLE_FLAG[@]}" \
      "${EXTRA[@]}" \
      2>&1 | tee -a "${LOG}"
done

echo "=== C′ no-AWR LATENCY DONE $(date -Iseconds) ===" | tee -a "${LOG}"
