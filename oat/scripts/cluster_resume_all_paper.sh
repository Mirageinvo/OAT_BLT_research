#!/usr/bin/env bash
# Relaunch all paper jobs after docker/tmux loss. Does NOT restart docker.
#
# GPU0: square train (resume) + can eval chain5
# GPU1: mt4 train (resume) + lift eval chain5
#
# Usage (inside container):
#   bash scripts/cluster_resume_all_paper.sh
set -euo pipefail
cd /workspace/oat
source .venv/bin/activate
export OAT_USE_UV_RUN=0

bash scripts/patch_robosuite_egl_assert.sh

SQUARE_RUN="${SQUARE_RUN_DIR:-output/20260707/102446_train_oatpolicy_square_N200}"
MT4_RUN="${MT4_RUN_DIR:-output/20260708/032431_train_oatpolicy_mw-mt4_N50}"

# Kill only our sessions (not unrelated tmux).
for s in policy_square_paper policy_mt4_paper eval_can_chain5 eval_lift_chain5; do
  tmux kill-session -t "${s}" 2>/dev/null || true
done
sleep 2

# Don't kill eval if already healthy — but we just killed sessions; clean stale children.
pkill -f "eval_policy_sim.*ep-1700" 2>/dev/null || true
pkill -f "eval_policy_sim.*ep-0600" 2>/dev/null || true
sleep 1

mkdir -p logs

tmux new-session -d -s policy_square_paper \
  "RESUME_RUN_DIR=${SQUARE_RUN} bash scripts/cluster_policy_square_paper.sh"

tmux new-session -d -s policy_mt4_paper \
  "RESUME_RUN_DIR=${MT4_RUN} bash scripts/cluster_policy_mt4_paper.sh"

tmux new-session -d -s eval_can_chain5 \
  "bash scripts/cluster_eval_can_chain5.sh"

tmux new-session -d -s eval_lift_chain5 \
  "bash scripts/cluster_eval_lift_chain5.sh"

echo "=== resumed all paper jobs ==="
tmux ls
echo "square resume: ${SQUARE_RUN}"
echo "mt4 resume:    ${MT4_RUN}"
