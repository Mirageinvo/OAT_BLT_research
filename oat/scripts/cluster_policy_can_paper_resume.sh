#!/usr/bin/env bash
set -euo pipefail
cd /workspace/oat
source .venv/bin/activate
export OAT_USE_UV_RUN=0 MUJOCO_GL=egl
export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
export RESUME_RUN_DIR="${RESUME_RUN_DIR:-output/20260706/173343_train_oatpolicy_can_N200}"
echo "=== can resume OAT8 eval | ${RESUME_RUN_DIR} ===" | tee -a logs/train_policy_can_paper_s42.log
bash scripts/cluster_policy_can_paper.sh 2>&1 | tee -a logs/train_policy_can_paper_s42.log
