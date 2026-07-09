#!/usr/bin/env bash
# MetaWorld MT4 full pipeline on GPU1: gen data (if needed) → fast policy train.
#
# GPU1 only (CUDA_VISIBLE_DEVICES=1). RoboMimic stays on GPU0.
#
# Usage:
#   SEED=0 bash scripts/cluster_mt4_pipeline.sh
#   SEED=0 bash scripts/cluster_mt4_pipeline.sh --skip-gen   # policy only (zarr must exist)
#
set -euo pipefail
cd /workspace/oat
source .venv/bin/activate

export OAT_USE_UV_RUN=0
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"

SEED="${SEED:-0}"
NUM_DEMO="${NUM_DEMO:-50}"
SKIP_GEN="${SKIP_GEN:-0}"
ZARR="data/metaworld/mt4_N${NUM_DEMO}.zarr"
PIPELINE_LOG="logs/mt4_pipeline_s${SEED}.log"
TOKENIZER_CKPT="${TOKENIZER_CKPT:-output/20260707/124135_train_oattok_mw-mt4_N50/checkpoints/ep-3830_mse-0.024.ckpt}"

mkdir -p logs data/metaworld

log() { echo "[$(date -Iseconds)] $*" | tee -a "${PIPELINE_LOG}"; }

for arg in "$@"; do
  case "${arg}" in
    --skip-gen) SKIP_GEN=1 ;;
  esac
done

log "=== MT4 pipeline seed=${SEED} GPU1 ==="
log "tokenizer=${TOKENIZER_CKPT}"

if [[ ! -f "${TOKENIZER_CKPT}" ]]; then
  log "ERROR: tokenizer checkpoint not found: ${TOKENIZER_CKPT}"
  exit 1
fi

zarr_ok() {
  python scripts/validate_metaworld_data.py "${ZARR}" --episodes-per-task "${NUM_DEMO}" >/dev/null 2>&1
}

need_gen=0
if [[ "${SKIP_GEN}" == "1" ]]; then
  log "skip-gen: expecting existing zarr at ${ZARR}"
  if ! zarr_ok; then
    log "ERROR: ${ZARR} missing or invalid (run without --skip-gen)"
    exit 1
  fi
  log "zarr OK"
else
  if zarr_ok; then
    log "zarr OK — skip generation (${ZARR})"
  else
    need_gen=1
  fi
fi

if [[ "${need_gen}" == "1" ]]; then
  if [[ -d "${ZARR}" ]]; then
    log "Removing incomplete zarr: ${ZARR}"
    rm -rf "${ZARR}"
  fi
  log "Generating ${ZARR} (${NUM_DEMO} demos × 4 tasks)..."
  bash scripts/prepare_metaworld_mt4.sh gen_data 2>&1 | tee -a "logs/gen_metaworld_mt4_N${NUM_DEMO}.log"
  if ! zarr_ok; then
    log "ERROR: generation finished but validation failed"
    exit 1
  fi
  log "Generation OK"
fi

log "Starting fast policy train (n_test=50, n_parallel_envs=8, rollout_start=200)..."
export TOKENIZER_CKPT
export SEED NUM_DEMO
exec bash scripts/cluster_policy_mt4_paper.sh 2>&1 | tee -a "${PIPELINE_LOG}"
