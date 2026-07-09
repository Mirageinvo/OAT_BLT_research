#!/usr/bin/env bash
# MetaWorld MT4 policy — fast train-time eval + top-k SR checkpoints (like RoboMimic).
#
# Defaults: eval @ ep 0,50,100,... | n_test=50 | n_parallel_envs=8 | num_workers=8 | GPU1
# Tokenizer frozen. Final paper eval: eval_metaworld_policy.sh @ n_test=250.
#
# Resume:
#   RESUME_RUN_DIR=output/20260708/032431_train_oatpolicy_mw-mt4_N50 \
#     bash scripts/cluster_policy_mt4_paper.sh
set -euo pipefail
cd /workspace/oat
source .venv/bin/activate

export OAT_USE_UV_RUN=0
# shellcheck disable=SC1091
source scripts/cluster_gpu_env.sh 1
bash scripts/patch_robosuite_egl_assert.sh

SEED="${SEED:-0}"
NUM_DEMO="${NUM_DEMO:-50}"
ROLLOUT_EVERY="${ROLLOUT_EVERY:-50}"
ROLLOUT_START_EPOCH="${ROLLOUT_START_EPOCH:-0}"
N_TEST="${N_TEST:-50}"
N_PARALLEL_ENVS="${N_PARALLEL_ENVS:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
TOKENIZER_CKPT="${TOKENIZER_CKPT:-output/20260707/124135_train_oattok_mw-mt4_N50/checkpoints/ep-3830_mse-0.024.ckpt}"
RESUME_RUN_DIR="${RESUME_RUN_DIR:-output/20260708/032431_train_oatpolicy_mw-mt4_N50}"
LOG="logs/train_policy_mt4_paper_s${SEED}.log"

if [[ ! -f "${TOKENIZER_CKPT}" ]]; then
  echo "ERROR: tokenizer checkpoint not found: ${TOKENIZER_CKPT}"
  exit 1
fi
if [[ ! -d "data/metaworld/mt4_N${NUM_DEMO}.zarr" ]]; then
  echo "ERROR: dataset missing: data/metaworld/mt4_N${NUM_DEMO}.zarr"
  exit 1
fi

echo "=== MT4 policy | seed=${SEED} | GPU1 | eval every ${ROLLOUT_EVERY} (from ep ${ROLLOUT_START_EPOCH}) ===" | tee "${LOG}"
echo "    tok=${TOKENIZER_CKPT}" | tee -a "${LOG}"
echo "    n_test=${N_TEST} n_parallel_envs=${N_PARALLEL_ENVS} num_workers=${NUM_WORKERS}" | tee -a "${LOG}"
if [[ -n "${RESUME_RUN_DIR}" ]]; then
  echo "    RESUME ${RESUME_RUN_DIR} (checkpoints/latest.ckpt)" | tee -a "${LOG}"
  HYDRA_EXTRA=("hydra.run.dir=${RESUME_RUN_DIR}" "training.resume=true")
else
  HYDRA_EXTRA=()
fi

HYDRA_FULL_ERROR=1 accelerate launch \
  --num_machines 1 --num_processes 1 \
  scripts/run_workspace.py \
  --config-name=train_oatpolicy \
  task/policy=metaworld/mt4 \
  task.policy.lazy_eval=false \
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
