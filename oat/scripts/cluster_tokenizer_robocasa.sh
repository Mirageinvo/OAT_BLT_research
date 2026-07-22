#!/usr/bin/env bash
# RoboCasa single-task OAT tokenizer (paper HP). Needs G0 zarr only — not G0b.
#
#   TASK=close_drawer GPU=1 bash scripts/cluster_tokenizer_robocasa.sh
# Optional: RESUME_RUN_DIR=output/.../train_oattok_close_drawer_N200
#
# Protocol: oat/ROBOCASA.md §2. Policy still blocked on G0b.
set -euo pipefail
cd /workspace/oat
source .venv/bin/activate

export OAT_USE_UV_RUN=0
GPU="${GPU:-1}"
# shellcheck disable=SC1091
source scripts/cluster_gpu_env.sh "${GPU}"

TASK="${TASK:?Set TASK (close_drawer|coffee_press_button|turn_off_microwave|turn_off_sink_faucet)}"
SEED="${SEED:-0}"
NUM_DEMO="${NUM_DEMO:-200}"
NUM_WORKERS="${NUM_WORKERS:-8}"
ZARR="data/robocasa/${TASK}_N${NUM_DEMO}.zarr"
DATE_TAG="$(date +%Y%m%d)"
TIME_TAG="$(date +%H%M%S)"
RUN_BASENAME="train_oattok_${TASK}_N${NUM_DEMO}"
DEFAULT_RUN_DIR="output/${DATE_TAG}/${TIME_TAG}_${RUN_BASENAME}"
RUN_DIR="${RESUME_RUN_DIR:-${DEFAULT_RUN_DIR}}"
LOG="logs/${RUN_BASENAME}_s${SEED}.log"

case "${TASK}" in
  close_drawer|coffee_press_button|turn_off_microwave|turn_off_sink_faucet) ;;
  *) echo "ERROR: unknown TASK=${TASK}"; exit 2 ;;
esac

if [[ ! -d "${ZARR}" ]]; then
  echo "ERROR: G0 zarr missing: ${ZARR}"
  exit 1
fi

mkdir -p logs "$(dirname "${RUN_DIR}")"
echo "=== robocasa tokenizer | task=${TASK} | seed=${SEED} | gpu=${GPU} | Da=12 ===" | tee "${LOG}"
echo "zarr=${ZARR} run_dir=${RUN_DIR}" | tee -a "${LOG}"

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
  "task/tokenizer=robocasa/${TASK}" \
  seed="${SEED}" \
  training.num_demo="${NUM_DEMO}" \
  training.seed="${SEED}" \
  dataloader.num_workers="${NUM_WORKERS}" \
  val_dataloader.num_workers="${NUM_WORKERS}" \
  checkpoint.topk.k=3 \
  checkpoint.topk.monitor_key=test_reconst_mse \
  logging.mode=disabled \
  "${HYDRA_EXTRA[@]}" \
  "$@" \
  2>&1 | tee -a "${LOG}"
