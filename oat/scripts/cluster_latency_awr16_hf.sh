#!/usr/bin/env bash
# Table C′ AWR16 latency from https://huggingface.co/Mirageinv/AWR
# Disk-safe: download ONE ckpt → measure → DELETE → next.
#
# HF currently has (2026-07-28):
#   robomimic_{can,lift,square}_awr_bon16_e100.ckpt
#   robocasa_coffee_press_button_awr_bon16_e100.ckpt
# MetaWorld: no AWR16 on HF → skip (AWR8 already in locked latency_fair_kv.json).
#
# Usage (inside docker /workspace/oat):
#   OAT_GIT_COMMIT=... GPU=0 bash scripts/cluster_latency_awr16_hf.sh
set -euo pipefail
cd /workspace/oat
mkdir -p logs /tmp/awr16_stage

GPU="${GPU:-0}"
REPS="${REPS:-10}"
TRIALS="${TRIALS:-8}"
WARMUP="${WARMUP:-20}"
HF_REPO="${HF_REPO:-Mirageinv/AWR}"
STAGE="${STAGE:-/tmp/awr16_stage}"
LOG="logs/latency_awr16_hf_gpu${GPU}.log"
RUNTIME_PY="${RUNTIME_PY:-/workspace/oat/_runtime_uv_python/cpython-3.10.20-linux-x86_64-gnu/bin/python3.10}"

if [[ -z "${OAT_GIT_COMMIT:-}" ]]; then
  echo "ERROR: OAT_GIT_COMMIT unset" >&2
  exit 1
fi
export OAT_GIT_COMMIT OAT_GIT_BRANCH="${OAT_GIT_BRANCH:-robocasa}" OAT_GIT_DIRTY="${OAT_GIT_DIRTY:-1}"
export MUJOCO_GL=egl OAT_USE_UV_RUN=0

# shellcheck disable=SC1091
source scripts/cluster_gpu_env.sh "${GPU}"

log() { echo "$*" | tee -a "${LOG}"; }

setup_python() {
  local venv="$1"
  if [[ -x "${venv}/bin/python" ]] && "${venv}/bin/python" -c "import hydra" 2>/dev/null; then
    # shellcheck disable=SC1091
    source "${venv}/bin/activate"
    PYTHON="${venv}/bin/python"
  else
    log "[awr16] ${venv}/bin/python broken — RUNTIME_PY + site-packages"
    export VIRTUAL_ENV
    VIRTUAL_ENV="$(cd "${venv}" && pwd)"
    export PATH="${VIRTUAL_ENV}/bin:${PATH}"
    export PYTHONPATH="/workspace/oat/_runtime_venv_boot:${VIRTUAL_ENV}/lib/python3.10/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
    PYTHON="${RUNTIME_PY}"
  fi
}

cleanup_stage() {
  rm -rf "${STAGE:?}"/*
}

hf_download() {
  local file="$1"
  local HF_BIN=""
  if [[ -x /home/askhabaliev_gs/.local/bin/hf ]]; then
    HF_BIN=/home/askhabaliev_gs/.local/bin/hf
  elif [[ -x /opt/conda/bin/hf ]]; then
    HF_BIN=/opt/conda/bin/hf
  else
    HF_BIN="$(command -v hf || true)"
  fi
  if [[ -z "${HF_BIN}" ]]; then
    log "ERROR: hf CLI not found"
    exit 1
  fi
  # Avoid .venv/bin/hf (broken shebang → host uv python)
  local PATH_NO_VENV
  PATH_NO_VENV="$(echo "${PATH}" | tr ':' '\n' | grep -v '/\.venv' | paste -sd: -)"
  PATH="${PATH_NO_VENV}" "${HF_BIN}" download "${HF_REPO}" "${file}" --local-dir "${STAGE}"
}

n16_path_for() {
  local suite="$1"
  case "${suite}" in
    coffee_press_button|close_drawer|turn_off_sink_faucet|turn_off_microwave)
      echo "output/eval/matched_s10000/robocasa/${suite}/latency_fair_kv_n16.json"
      ;;
    *)
      echo "output/eval/matched_s10000/${suite}/latency_fair_kv_n16.json"
      ;;
  esac
}

has_awr16() {
  local out="$1"
  [[ -f "${out}" ]] || return 1
  python3 -c "import json,sys; m=json.load(open(sys.argv[1])).get('modes',{}); sys.exit(0 if 'awr16' in m else 1)" "${out}"
}

ALL_JOBS=(
  "can|robomimic_can_awr_bon16_e100.ckpt|.venv"
  "lift|robomimic_lift_awr_bon16_e100.ckpt|.venv"
  "square|robomimic_square_awr_bon16_e100.ckpt|.venv"
  "coffee_press_button|robocasa_coffee_press_button_awr_bon16_e100.ckpt|.venv_robocasa"
)

JOBS=()
for job in "${ALL_JOBS[@]}"; do
  IFS='|' read -r suite hf_file venv <<<"${job}"
  out="$(n16_path_for "${suite}")"
  if has_awr16 "${out}"; then
    log "[skip] ${suite} already has awr16"
    continue
  fi
  JOBS+=("${job}")
done

log "=== AWR16 HF latency $(date -Iseconds) gpu=${CUDA_VISIBLE_DEVICES} repo=${HF_REPO} ==="
log "disk before: $(df -h /workspace/oat | tail -1)"
log "remaining jobs: ${#JOBS[@]}"
if [[ ${#JOBS[@]} -eq 0 ]]; then
  log "nothing left to measure"
  exit 0
fi
cleanup_stage

for job in "${JOBS[@]}"; do
  IFS='|' read -r suite hf_file venv <<<"${job}"
  log ""
  log "[AWR16] ${suite} ← ${hf_file} $(date -Iseconds)"
  cleanup_stage
  df -h /workspace/oat /tmp | tee -a "${LOG}"

  log "[dl] hf download ${HF_REPO} ${hf_file}"
  hf_download "${hf_file}" 2>&1 | tee -a "${LOG}"
  ckpt="${STAGE}/${hf_file}"
  [[ -f "${ckpt}" ]] || { log "ERROR missing ${ckpt}"; exit 1; }
  log "[dl] ok size=$(du -h "${ckpt}" | awk '{print $1}')"

  setup_python "${venv}"
  bash scripts/patch_robosuite_egl_assert.sh || true

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl \
    OAT_GIT_COMMIT="${OAT_GIT_COMMIT}" OAT_GIT_BRANCH="${OAT_GIT_BRANCH}" OAT_GIT_DIRTY="${OAT_GIT_DIRTY}" \
    PYTHONPATH="${PYTHONPATH:-}" VIRTUAL_ENV="${VIRTUAL_ENV:-}" \
    "${PYTHON}" scripts/measure_latency_paper.py \
      --suite "${suite}" \
      -d "${OAT_DEVICE}" \
      --reps "${REPS}" \
      --trials "${TRIALS}" \
      --warmup "${WARMUP}" \
      --fair_kv \
      --skip_awr \
      --no_single \
      --bon_ns "" \
      --awr16_ckpt "${ckpt}" \
      2>&1 | tee -a "${LOG}"

  log "[rm] deleting ${ckpt}"
  rm -f "${ckpt}"
  cleanup_stage
  log "[rm] disk now: $(df -h /workspace/oat | tail -1)"
done

log "=== AWR16 HF latency ALL DONE $(date -Iseconds) ==="
