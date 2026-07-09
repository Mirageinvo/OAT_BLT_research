#!/usr/bin/env bash
set -euo pipefail
cd /workspace/oat
source .venv/bin/activate
export OAT_USE_UV_RUN=0 MUJOCO_GL=egl SEED=42 NUM_PROCESSES=1
python scripts/validate_robomimic_data.py
export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
export POLICY_LOG=logs/train_policy_lift_paper_s42.log
CUDA_VISIBLE_DEVICES=0 bash scripts/run_policy_robomimic_paper.sh lift \
  output/20260704/203215_train_oattok_lift_N200/checkpoints/ep-1970_mse-0.006.ckpt
