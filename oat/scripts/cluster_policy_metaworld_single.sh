#!/usr/bin/env bash
# Single-task MetaWorld policy training (paper-style train-time eval, no fast shortcuts).
#
# Required:
#   TASK=box-close|coffee-pull|disassemble|stick-pull
# Optional:
#   TOKENIZER_CKPT=...
#   RESUME_RUN_DIR=output/.../train_oatpolicy_mw-box-close_st_N50
set -euo pipefail
cd /workspace/oat
source .venv/bin/activate

export OAT_USE_UV_RUN=0
GPU="${GPU:-0}"
# shellcheck disable=SC1091
source scripts/cluster_gpu_env.sh "${GPU}"
bash scripts/patch_robosuite_egl_assert.sh

TASK="${TASK:?Set TASK (box-close|coffee-pull|disassemble|stick-pull)}"
SEED="${SEED:-0}"
NUM_DEMO="${NUM_DEMO:-50}"
ROLLOUT_EVERY="${ROLLOUT_EVERY:-200}"
ROLLOUT_START_EPOCH="${ROLLOUT_START_EPOCH:-0}"
N_TEST="${N_TEST:-250}"
N_PARALLEL_ENVS="${N_PARALLEL_ENVS:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DATE_TAG="$(date +%Y%m%d)"
TIME_TAG="$(date +%H%M%S)"
RUN_BASENAME="train_oatpolicy_mw-${TASK}_st_N${NUM_DEMO}"
DEFAULT_RUN_DIR="output/${DATE_TAG}/${TIME_TAG}_${RUN_BASENAME}"
RUN_DIR="${RESUME_RUN_DIR:-${DEFAULT_RUN_DIR}}"
LOG="logs/${RUN_BASENAME}_s${SEED}.log"

if [[ -z "${TOKENIZER_CKPT:-}" ]]; then
  echo "ERROR: set TOKENIZER_CKPT before running policy training."
  exit 1
fi
if [[ ! -f "${TOKENIZER_CKPT}" ]]; then
  echo "ERROR: tokenizer checkpoint not found: ${TOKENIZER_CKPT}"
  exit 1
fi
if [[ ! -d "data/metaworld/${TASK}_N${NUM_DEMO}.zarr" ]]; then
  echo "ERROR: dataset missing: data/metaworld/${TASK}_N${NUM_DEMO}.zarr"
  exit 1
fi

mkdir -p logs "$(dirname "${RUN_DIR}")"
echo "=== single-task policy | task=${TASK} | seed=${SEED} | gpu=${GPU} ===" | tee "${LOG}"
echo "run_dir=${RUN_DIR}" | tee -a "${LOG}"
echo "tok=${TOKENIZER_CKPT}" | tee -a "${LOG}"
echo "rollout_every=${ROLLOUT_EVERY} n_test=${N_TEST} n_parallel_envs=${N_PARALLEL_ENVS}" | tee -a "${LOG}"

HYDRA_EXTRA=("hydra.run.dir=${RUN_DIR}")
if [[ -n "${RESUME_RUN_DIR:-}" ]]; then
  HYDRA_EXTRA+=("training.resume=true")
else
  HYDRA_EXTRA+=("training.resume=false")
fi

HYDRA_FULL_ERROR=1 accelerate launch \
  --num_machines 1 --num_processes 1 \
  scripts/run_workspace.py \
  --config-name=train_oatpolicy \
  task/policy=metaworld/mt4 \
  task.policy.name="mw-${TASK}-st" \
  task.policy.task_name="${TASK}" \
  task.policy.lazy_eval=false \
  "task.policy.dataset.zarr_path=data/metaworld/${TASK}_N${NUM_DEMO}.zarr" \
  "task.policy.env_runner.task_name=${TASK}" \
  "policy.action_tokenizer.checkpoint=${TOKENIZER_CKPT}" \
  seed="${SEED}" \
  training.num_demo="${NUM_DEMO}" \
  training.rollout_every="${ROLLOUT_EVERY}" \
  training.rollout_start_epoch="${ROLLOUT_START_EPOCH}" \
  training.checkpoint_every="${ROLLOUT_EVERY}" \
  dataloader.num_workers="${NUM_WORKERS}" \
  val_dataloader.num_workers="${NUM_WORKERS}" \
  task.policy.env_runner.n_test="${N_TEST}" \
  task.policy.env_runner.n_parallel_envs="${N_PARALLEL_ENVS}" \
  checkpoint.topk.k=3 \
  checkpoint.topk.monitor_key=mean_success_rate \
  logging.mode=disabled \
  "${HYDRA_EXTRA[@]}" \
  "$@" \
  2>&1 | tee -a "${LOG}"
