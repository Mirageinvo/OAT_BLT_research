#!/usr/bin/env bash
# Train one policy until TIME_LIMIT_SEC (default 2.5h), frozen tokenizer, from scratch.
set -euo pipefail
cd "$(dirname "$0")/../.."

TASK="${1:?task}"
GPU="${2:-0}"
TIME_LIMIT_SEC="${TIME_LIMIT_SEC:-9000}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export CUDA_VISIBLE_DEVICES="${GPU}"

STAGE="${STAGE:-hackathon_assets}"
LOG_DIR="${LOG_DIR:-hackathon_output/logs}"
if [[ "${OAT_USE_UV_RUN:-1}" == "0" ]]; then
  RUNNER=()
elif command -v uv >/dev/null 2>&1; then
  RUNNER=(uv run)
else
  RUNNER=()
fi
NUM_DEMO="${NUM_DEMO:-200}"
NUM_EPOCHS="${NUM_EPOCHS:-10000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-15}"
# Hackathon POC: no in-train sim — one short eval after fit (eval_and_demo_one.sh)
ROLLOUT_EVERY="${ROLLOUT_EVERY:-200}"
ROLLOUT_START="${ROLLOUT_START:-100000}"
BATCH="${BATCH:-64}"

tok="${STAGE}/tokenizers/${TASK}.ckpt"
[[ -f "${tok}" ]] || { echo "Missing ${tok}"; exit 1; }

mkdir -p "${LOG_DIR}"
log="${LOG_DIR}/train_${TASK}.log"

{
  echo "=== TRAIN ${TASK} START $(date -Iseconds) gpu=${GPU} limit=${TIME_LIMIT_SEC}s ==="
  echo "tokenizer=${tok}"

  timeout -s INT "${TIME_LIMIT_SEC}" \
    env OAT_USE_UV_RUN=0 \
    "${RUNNER[@]}" accelerate launch --num_processes 1 \
    scripts/run_workspace.py \
    --config-name=train_oatpolicy \
    "task/policy=robomimic/${TASK}" \
    "policy.action_tokenizer.checkpoint=${tok}" \
    "training.num_demo=${NUM_DEMO}" \
    "training.num_epochs=${NUM_EPOCHS}" \
    "training.rollout_every=${ROLLOUT_EVERY}" \
    "training.rollout_start_epoch=${ROLLOUT_START}" \
    "training.checkpoint_every=${CHECKPOINT_EVERY}" \
    "dataloader.batch_size=${BATCH}" \
    "val_dataloader.batch_size=${BATCH}" \
    "checkpoint.topk.k=1" \
    "logging.mode=disabled" \
    "task.policy.lazy_eval=true" \
    "task.policy.env_runner.n_parallel_envs=2" \
    || true

  exit_code=$?
  echo "=== TRAIN ${TASK} STOP $(date -Iseconds) exit=${exit_code} (124=timeout) ==="
} 2>&1 | tee "${log}"
