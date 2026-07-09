#!/usr/bin/env bash
# Can policy on GPU1 alongside square tokenizer — lower n_parallel_envs to save VRAM.
set -euo pipefail
cd /workspace/oat
source .venv/bin/activate
export OAT_USE_UV_RUN=0 MUJOCO_GL=egl
export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
bash scripts/cluster_ensure_v100_torch.sh 2>/dev/null || true
LOG=logs/train_policy_can_paper_s42.log
CAN_TOK=output/20260705/210939_train_oattok_can_N200/checkpoints/ep-0520_mse-0.005.ckpt
echo "=== can paper-default seed=42 | GPU0 (shared with lift) | n_parallel_envs=2 ===" | tee "${LOG}"
# GPU1 + EGL is broken in this docker (CUDA_VISIBLE_DEVICES remaps vs MUJOCO_EGL).
# Share GPU0 with lift — ~5+5 GB fits on 32 GB V100.
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
HYDRA_EXTRA=()
if [[ -n "${RESUME_RUN_DIR:-}" ]]; then
  HYDRA_EXTRA+=("hydra.run.dir=${RESUME_RUN_DIR}")
fi
HYDRA_FULL_ERROR=1 accelerate launch \
  --num_machines 1 --num_processes 1 \
  scripts/run_workspace.py \
  --config-name=train_oatpolicy \
  task/policy=robomimic/can \
  task.policy.lazy_eval=false \
  "policy.action_tokenizer.checkpoint=${CAN_TOK}" \
  training.num_demo=200 \
  training.rollout_every=100 \
  training.seed=42 \
  task.policy.env_runner.n_parallel_envs=2 \
  checkpoint.topk.k=3 \
  checkpoint.topk.monitor_key=mean_success_rate \
  logging.mode=disabled \
  "${HYDRA_EXTRA[@]}" \
  2>&1 | tee -a "${LOG}"
