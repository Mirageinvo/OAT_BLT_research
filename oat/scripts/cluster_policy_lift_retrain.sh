#!/usr/bin/env bash
# Lift policy RETRAIN for paper (target ~90% chain5 / matched base — NOT Table VI 99.2%).
#
# Why retrain: old ep-0600 train-eval 92% (n_test=50) → chain5 only 83.6%. TopK was noisy.
# Plan: keep frozen tokenizer; new train seed; larger train-time n_test for TopK.
#
# Usage (inside oat docker):
#   GPU=0 SEED=7 bash scripts/cluster_policy_lift_retrain.sh
#
# Paper path = SAME as Can/Square (NOT chain5):
#   pick TopK ckpt → matched Wave1 baseline+BoN @ s10000 → Wave2 AWR if BoN ok.
#   SUITE=lift BASE_CKPT=<topk> GPU=0 bash scripts/cluster_matched_triplet.sh
# Early stop hint: train-time TopK SR (n_test=100) ≳0.90 — then run matched, don't wait 5k ep.
# Gate on MATCHED baseline @10000: ≳~0.88–0.90 → keep in Table P; else SEED=13 once;
#   still <~0.85 → omit/footnote. chain5 = lab only, never paper comparator.

set -euo pipefail
cd /workspace/oat
source .venv/bin/activate

export OAT_USE_UV_RUN=0
export MUJOCO_GL=egl
export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"

GPU="${GPU:-0}"
SEED="${SEED:-7}"
NUM_DEMO="${NUM_DEMO:-200}"
ROLLOUT_EVERY="${ROLLOUT_EVERY:-100}"
# Larger than old n_test=50 so TopK closer to chain5/matched; 100 = cost compromise (250 = MW-style, ~5× slower eval).
N_TEST="${N_TEST:-100}"
N_PARALLEL_ENVS="${N_PARALLEL_ENVS:-4}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"

TOK="${TOKENIZER_CKPT:-output/20260704/203215_train_oattok_lift_N200/checkpoints/ep-1970_mse-0.006.ckpt}"
LOG="${POLICY_LOG:-logs/train_policy_lift_retrain_s${SEED}_n${N_TEST}.log}"

if [[ ! -f "${TOK}" ]]; then
  echo "ERROR: tokenizer missing: ${TOK}"
  exit 1
fi
if [[ ! -d data/robomimic/lift_N200.zarr ]]; then
  echo "ERROR: missing data/robomimic/lift_N200.zarr"
  exit 1
fi

mkdir -p logs
export CUDA_VISIBLE_DEVICES="${GPU}"
export SEED NUM_DEMO ROLLOUT_EVERY NUM_PROCESSES POLICY_LOG="${LOG}"

echo "=== Lift RETRAIN | seed=${SEED} n_test=${N_TEST} rollout_every=${ROLLOUT_EVERY} gpu=${GPU} ===" | tee "${LOG}"
echo "  tok=${TOK}" | tee -a "${LOG}"
echo "  target: matched_s10000 baseline ~90% (same proto as Can/Square; not 99.2)" | tee -a "${LOG}"
echo "  started $(date -Iseconds)" | tee -a "${LOG}"

python scripts/validate_robomimic_data.py 2>&1 | tee -a "${LOG}"

if [[ -f scripts/cluster_ensure_v100_torch.sh ]]; then
  bash scripts/cluster_ensure_v100_torch.sh
fi

HYDRA_FULL_ERROR=1 accelerate launch \
  --num_machines 1 \
  --num_processes "${NUM_PROCESSES}" \
  scripts/run_workspace.py \
  --config-name=train_oatpolicy \
  task/policy=robomimic/lift \
  task.policy.lazy_eval=false \
  "task.policy.env_runner.n_test=${N_TEST}" \
  "task.policy.env_runner.n_parallel_envs=${N_PARALLEL_ENVS}" \
  "policy.action_tokenizer.checkpoint=${TOK}" \
  training.num_demo="${NUM_DEMO}" \
  training.rollout_every="${ROLLOUT_EVERY}" \
  training.seed="${SEED}" \
  seed="${SEED}" \
  checkpoint.topk.k=3 \
  checkpoint.topk.monitor_key=mean_success_rate \
  logging.mode=disabled \
  2>&1 | tee -a "${LOG}"

echo "=== Lift RETRAIN finished $(date -Iseconds) ===" | tee -a "${LOG}"
echo "NEXT: pick best top-k → matched Wave1/2 @ s10000 (Can/Square recipe; NOT chain5)" | tee -a "${LOG}"
