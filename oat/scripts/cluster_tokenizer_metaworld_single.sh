#!/usr/bin/env bash
# Single-task MetaWorld tokenizer training (top-1 by reconstruction).
#
# Required:
#   TASK=box-close|coffee-pull|disassemble|stick-pull
# Optional:
#   RESUME_RUN_DIR=output/.../train_oattok_mw-box-close_st_N50
set -euo pipefail
cd /workspace/oat
source .venv/bin/activate

export OAT_USE_UV_RUN=0
GPU="${GPU:-0}"
# shellcheck disable=SC1091
source scripts/cluster_gpu_env.sh "${GPU}"

TASK="${TASK:?Set TASK (box-close|coffee-pull|disassemble|stick-pull)}"
SEED="${SEED:-0}"
NUM_DEMO="${NUM_DEMO:-50}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DATE_TAG="$(date +%Y%m%d)"
TIME_TAG="$(date +%H%M%S)"
RUN_BASENAME="train_oattok_mw-${TASK}_st_N${NUM_DEMO}"
DEFAULT_RUN_DIR="output/${DATE_TAG}/${TIME_TAG}_${RUN_BASENAME}"
RUN_DIR="${RESUME_RUN_DIR:-${DEFAULT_RUN_DIR}}"
LOG="logs/${RUN_BASENAME}_s${SEED}.log"

if [[ ! -d "data/metaworld/${TASK}_N${NUM_DEMO}.zarr" ]]; then
  echo "ERROR: dataset missing: data/metaworld/${TASK}_N${NUM_DEMO}.zarr"
  exit 1
fi

mkdir -p logs "$(dirname "${RUN_DIR}")"
echo "=== single-task tokenizer | task=${TASK} | seed=${SEED} | gpu=${GPU} ===" | tee "${LOG}"
echo "run_dir=${RUN_DIR}" | tee -a "${LOG}"

HYDRA_EXTRA=("hydra.run.dir=${RUN_DIR}")
if [[ -n "${RESUME_RUN_DIR:-}" ]]; then
  HYDRA_EXTRA+=("training.resume=true")
else
  HYDRA_EXTRA+=("training.resume=false")
fi

HYDRA_FULL_ERROR=1 accelerate launch \
  --num_machines 1 --num_processes 1 \
  scripts/run_workspace.py \
  --config-name=train_oattok \
  task/tokenizer=metaworld/mt4 \
  task.tokenizer.name="mw-${TASK}-st" \
  task.tokenizer.task_name="${TASK}" \
  "task.tokenizer.dataset.zarr_path=data/metaworld/${TASK}_N${NUM_DEMO}.zarr" \
  seed="${SEED}" \
  training.num_demo="${NUM_DEMO}" \
  dataloader.num_workers="${NUM_WORKERS}" \
  val_dataloader.num_workers="${NUM_WORKERS}" \
  checkpoint.topk.k=1 \
  checkpoint.topk.monitor_key=test_reconst_mse \
  logging.mode=disabled \
  "${HYDRA_EXTRA[@]}" \
  "$@" \
  2>&1 | tee -a "${LOG}"
