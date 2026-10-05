#!/usr/bin/env bash
# Reproduce paper baseline SR (matched_s10000 protocol) with HF policy ckpts.
set -euo pipefail
cd "$(dirname "$0")/../.."

export MUJOCO_GL="${MUJOCO_GL:-egl}"
STAGE="${STAGE:-hackathon_assets}"
OUT="${OUT:-hackathon_output/eval/matched_s10000}"

NUM_EXP="${NUM_EXP:-5}"
N_TEST="${N_TEST:-50}"
TEST_START_SEED="${TEST_START_SEED:-10000}"

eval_one() {
  local task="$1"
  local ckpt="${STAGE}/policies/${task}.ckpt"
  if [[ ! -f "${ckpt}" ]]; then
    echo "Missing ${ckpt}. Run 00_download_assets.sh first."
    exit 1
  fi
  local out="${OUT}/${task}/baseline_n5"
  echo "========== EVAL ${task} (${ckpt}) =========="
  echo "  protocol: test_start_seed=${TEST_START_SEED} n_test=${N_TEST} num_exp=${NUM_EXP} OAT8"
  MUJOCO_GL=egl uv run scripts/eval_policy_sim.py \
    --checkpoint "${ckpt}" \
    --output_dir "${out}" \
    --num_exp "${NUM_EXP}" \
    --n_test "${N_TEST}" \
    --test_start_seed "${TEST_START_SEED}" \
    --entropy_threshold 0 \
    --use_k_tokens 8
}

for task in lift can square; do
  eval_one "${task}"
done

echo "Eval done -> ${OUT}/"
echo "Compare with paper summaries in hackathon_output/eval/matched_s10000/*/summary.json (from HF)"
