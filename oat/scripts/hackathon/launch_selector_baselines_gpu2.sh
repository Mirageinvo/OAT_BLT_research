#!/usr/bin/env bash
# Launch selector-baseline evals on GPU 2 inside tmux (sequential queue).
# Usage (inside container, /workspace/oat):
#   bash scripts/hackathon/launch_selector_baselines_gpu2.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_EGL_DEVICE_ID=0
export OAT_USE_UV_RUN=0

OUT_ROOT="${OUT_ROOT:-output/eval/aamas27_selector_baselines}"
N_PARALLEL="${N_PARALLEL:-8}"
SESSION="${SESSION:-sel_gpu2}"
CKPT_CAN="${CKPT_CAN:?set CKPT_CAN to paper base can ckpt}"

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/can"

run_one() {
  local signal="$1"
  local out="${OUT_ROOT}/can/${signal}_n8"
  local log="${OUT_ROOT}/logs/can_${signal}_n8.log"
  echo "=== $(date -Iseconds) START can ${signal} → ${out} ===" | tee -a "${log}"
  python scripts/eval_policy_sim.py \
    --checkpoint "${CKPT_CAN}" \
    --output_dir "${out}" \
    --force \
    --device cuda:0 \
    --num_exp 5 \
    --n_test 50 \
    --test_start_seed 10000 \
    --n_parallel_envs "${N_PARALLEL}" \
    --use_k_tokens 8 \
    --entropy_threshold 0 \
    --temperature 1.0 \
    --topk 10 \
    --bon_free 8 \
    --bon_signal "${signal}" \
    --selector_seed 0 \
    2>&1 | tee -a "${log}"
  echo "=== $(date -Iseconds) DONE can ${signal} ===" | tee -a "${log}"
}

# Prefer attach to existing session or create
if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${SESSION}" -n queue
fi

# Queue: smoke-ish first (vote sanity) then required baselines
tmux send-keys -t "${SESSION}:0" "cd /workspace/oat && source .venv/bin/activate 2>/dev/null; export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CKPT_CAN='${CKPT_CAN}' OUT_ROOT='${OUT_ROOT}' N_PARALLEL=${N_PARALLEL}" C-m
tmux send-keys -t "${SESSION}:0" "bash -lc 'run_one(){ signal=\$1; out=${OUT_ROOT}/can/\${signal}_n8; log=${OUT_ROOT}/logs/can_\${signal}_n8.log; mkdir -p \"\$out\" \"\$(dirname \$log)\"; echo START \$signal; python scripts/eval_policy_sim.py -c \"\$CKPT_CAN\" -o \"\$out\" --force -d cuda:0 -n 5 --n_test 50 --test_start_seed 10000 --n_parallel_envs ${N_PARALLEL} --use_k_tokens 8 --entropy_threshold 0 --temperature 1.0 --topk 10 --bon_free 8 --bon_signal \$signal --selector_seed 0 2>&1 | tee -a \$log; echo DONE \$signal; }; run_one vote; run_one random; run_one medoid; run_one max_likelihood'" C-m

echo "tmux session: ${SESSION}  (attach: tmux attach -t ${SESSION})"
echo "GPU: ${CUDA_VISIBLE_DEVICES}  n_parallel_envs=${N_PARALLEL}"
