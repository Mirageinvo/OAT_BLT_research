#!/usr/bin/env bash
# Paper-default RoboMimic policy train (no aggressive profile).
# Sim-eval every rollout_every epochs; top-3 by mean_success_rate.
#
# Usage:
#   cd oat && source .venv/bin/activate
#   export MUJOCO_GL=egl OAT_USE_UV_RUN=0
#   export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
#
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_policy_robomimic_paper.sh lift \
#     output/20260704/203215_train_oattok_lift_N200/checkpoints/ep-1970_mse-0.006.ckpt
#
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_policy_robomimic_paper.sh can \
#     output/20260705/210939_train_oattok_can_N200/checkpoints/ep-0520_mse-0.005.ckpt

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

TASK="${1:?lift|can|square}"
TOKENIZER_CKPT="${2:?path to tokenizer .ckpt}"
SEED="${SEED:-42}"
NUM_DEMO="${NUM_DEMO:-200}"
ROLLOUT_EVERY="${ROLLOUT_EVERY:-100}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
LOG="${POLICY_LOG:-logs/train_policy_${TASK}_paper_s${SEED}.log}"
HYDRA_EXTRA=()
if [[ -n "${RESUME_RUN_DIR:-}" ]]; then
  HYDRA_EXTRA+=("hydra.run.dir=${RESUME_RUN_DIR}")
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OAT_USE_UV_RUN="${OAT_USE_UV_RUN:-0}"
if [[ -d "${HOME}/.mujoco/mujoco210/bin" ]]; then
  export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
fi

if [[ -f scripts/cluster_ensure_v100_torch.sh ]]; then
  bash scripts/cluster_ensure_v100_torch.sh
fi

mkdir -p logs
echo "=== paper-default policy | task=${TASK} seed=${SEED} tok=${TOKENIZER_CKPT} ===" | tee "${LOG}"
echo "  GPU=${CUDA_VISIBLE_DEVICES:-all} | rollout_every=${ROLLOUT_EVERY} | topk SR=3" | tee -a "${LOG}"
echo "  NO aggressive profile (batch=256, lr=5e-5/1e-5)" | tee -a "${LOG}"

HYDRA_FULL_ERROR=1 accelerate launch \
  --num_machines 1 \
  --num_processes "${NUM_PROCESSES}" \
  scripts/run_workspace.py \
  --config-name=train_oatpolicy \
  "task/policy=robomimic/${TASK}" \
  task.policy.lazy_eval=false \
  "policy.action_tokenizer.checkpoint=${TOKENIZER_CKPT}" \
  training.num_demo="${NUM_DEMO}" \
  training.rollout_every="${ROLLOUT_EVERY}" \
  training.seed="${SEED}" \
  checkpoint.topk.k=3 \
  checkpoint.topk.monitor_key=mean_success_rate \
  logging.mode=disabled \
  "${HYDRA_EXTRA[@]}" \
  2>&1 | tee -a "${LOG}"
