#!/usr/bin/env bash
# Resume Lift retrain in the SAME Hydra run dir (preserves top-k + logs for paper).
# Loads checkpoints/latest.ckpt; TopK continues in-place.
#
# Usage (docker):
#   bash scripts/cluster_policy_lift_retrain_resume.sh
#
# Anti-leak: train TopK uses selection pool (env_runner default seed 1000).
# Paper numbers ONLY from matched_s10000 (test_start_seed=10000), not train SR.
set -euo pipefail
cd /workspace/oat
source .venv/bin/activate

export OAT_USE_UV_RUN=0
export MUJOCO_GL=egl
export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"

GPU="${GPU:-0}"
SEED="${SEED:-7}"
N_TEST="${N_TEST:-100}"
N_PARALLEL_ENVS="${N_PARALLEL_ENVS:-4}"
RUN_DIR="${RESUME_RUN_DIR:-output/20260719/144024_train_oatpolicy_lift_N200}"
TOK="${TOKENIZER_CKPT:-output/20260704/203215_train_oattok_lift_N200/checkpoints/ep-1970_mse-0.006.ckpt}"
LOG="${POLICY_LOG:-logs/train_policy_lift_retrain_s${SEED}_n${N_TEST}.log}"

# shellcheck disable=SC1091
source scripts/cluster_gpu_env.sh "${GPU}"
bash scripts/patch_robosuite_egl_assert.sh

[[ -f "${RUN_DIR}/checkpoints/latest.ckpt" ]] || {
  echo "ERROR: missing ${RUN_DIR}/checkpoints/latest.ckpt"
  exit 1
}
[[ -f "${TOK}" ]] || {
  echo "ERROR: missing tokenizer ${TOK}"
  exit 1
}
[[ -f data/robomimic/hdf5_datasets/lift_mh_image.hdf5 ]] || {
  echo "ERROR: missing lift HDF5 (needed for train-time rollout)"
  exit 1
}

mkdir -p logs
{
  echo ""
  echo "=== Lift RETRAIN RESUME | seed=${SEED} n_test=${N_TEST} gpu=${GPU} ==="
  echo "  run_dir=${RUN_DIR}"
  echo "  tok=${TOK}"
  echo "  loads checkpoints/latest.ckpt; appends to this log; keeps existing top-k"
  echo "  anti-leak: paper report = matched_s10000 only (not train TopK SR)"
  echo "  started $(date -Iseconds)"
} | tee -a "${LOG}"

HYDRA_FULL_ERROR=1 accelerate launch \
  --num_machines 1 \
  --num_processes 1 \
  scripts/run_workspace.py \
  --config-name=train_oatpolicy \
  task/policy=robomimic/lift \
  task.policy.lazy_eval=false \
  "task.policy.env_runner.n_test=${N_TEST}" \
  "task.policy.env_runner.n_parallel_envs=${N_PARALLEL_ENVS}" \
  "policy.action_tokenizer.checkpoint=${TOK}" \
  training.num_demo=200 \
  training.rollout_every=100 \
  training.seed="${SEED}" \
  seed="${SEED}" \
  training.resume=true \
  "hydra.run.dir=${RUN_DIR}" \
  checkpoint.topk.k=3 \
  checkpoint.topk.monitor_key=mean_success_rate \
  logging.mode=disabled \
  2>&1 | tee -a "${LOG}"

echo "=== Lift RETRAIN RESUME finished $(date -Iseconds) ===" | tee -a "${LOG}"
