#!/usr/bin/env bash
# Square policy paper-default (or resume) on the cluster.
#
# Fresh:
#   bash scripts/cluster_policy_square_paper.sh
# Resume from latest.ckpt in an existing Hydra run dir:
#   RESUME_RUN_DIR=output/20260707/102446_train_oatpolicy_square_N200 \
#     bash scripts/cluster_policy_square_paper.sh
set -euo pipefail
cd /workspace/oat
source .venv/bin/activate

export OAT_USE_UV_RUN=0
# shellcheck disable=SC1091
source scripts/cluster_gpu_env.sh 0
bash scripts/patch_robosuite_egl_assert.sh

LOG=logs/train_policy_square_paper_s42.log
SQUARE_TOK="${SQUARE_TOK:-output/20260706/005048_train_oattok_square_N200/checkpoints/ep-0690_mse-0.004.ckpt}"
# Fresh by default; set RESUME_RUN_DIR explicitly to resume an existing Hydra run.
RESUME_RUN_DIR="${RESUME_RUN_DIR:-}"

echo "=== square paper-default seed=42 | GPU0 | n_parallel_envs=2 | tok=${SQUARE_TOK} ===" | tee -a "${LOG}"
if [[ -n "${RESUME_RUN_DIR}" ]]; then
  echo "=== RESUME ${RESUME_RUN_DIR} (loads checkpoints/latest.ckpt) ===" | tee -a "${LOG}"
fi

HYDRA_EXTRA=()
if [[ -n "${RESUME_RUN_DIR}" ]]; then
  HYDRA_EXTRA+=("hydra.run.dir=${RESUME_RUN_DIR}" "training.resume=true")
fi

HYDRA_FULL_ERROR=1 accelerate launch \
  --num_machines 1 --num_processes 1 \
  scripts/run_workspace.py \
  --config-name=train_oatpolicy \
  task/policy=robomimic/square \
  task.policy.lazy_eval=false \
  "policy.action_tokenizer.checkpoint=${SQUARE_TOK}" \
  training.num_demo=200 \
  training.rollout_every=100 \
  training.seed=42 \
  task.policy.env_runner.n_parallel_envs=2 \
  checkpoint.topk.k=3 \
  checkpoint.topk.monitor_key=mean_success_rate \
  logging.mode=disabled \
  "${HYDRA_EXTRA[@]}" \
  "$@" \
  2>&1 | tee -a "${LOG}"
