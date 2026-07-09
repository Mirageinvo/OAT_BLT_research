#!/usr/bin/env bash
# Restart lift/can paper policy runs with OAT8 train-eval fix, resume from existing run dirs.
set -euo pipefail
cd /workspace/oat
source .venv/bin/activate
export OAT_USE_UV_RUN=0 MUJOCO_GL=egl
export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"

pkill -f "run_workspace.py.*robomimic/lift" 2>/dev/null || true
pkill -f "run_workspace.py.*robomimic/can" 2>/dev/null || true
sleep 2
tmux kill-session -t policy_lift_paper 2>/dev/null || true
tmux kill-session -t policy_can_paper 2>/dev/null || true

LIFT_RUN="${LIFT_RUN_DIR:-output/20260706/163500_train_oatpolicy_lift_N200}"
CAN_RUN="${CAN_RUN_DIR:-output/20260706/173343_train_oatpolicy_can_N200}"
LIFT_TOK="${LIFT_TOK:-output/20260704/203215_train_oattok_lift_N200/checkpoints/ep-1970_mse-0.006.ckpt}"

tmux new-session -d -s policy_lift_paper "bash /workspace/oat/scripts/cluster_policy_lift_paper_resume.sh"
tmux new-session -d -s policy_can_paper "bash /workspace/oat/scripts/cluster_policy_can_paper_resume.sh"

echo "Restarted policy_lift_paper (resume ${LIFT_RUN}) and policy_can_paper (resume ${CAN_RUN})"
tmux ls
