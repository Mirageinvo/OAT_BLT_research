#!/usr/bin/env bash
# Launch PAPER matched baseline+BoN for all NOW suites in parallel (1 tmux each).
# Protocol: seed=10000, n_test=50, n_exp=5, OAT8, BoN N=8 vote, SKIP_AWR=1.
#
# From host (outside docker):
#   bash scripts/cluster_launch_matched_paper_wave.sh
#
# Inside docker /workspace/oat this also works if tmux is available on host wrappers —
# prefer host launch below.
set -euo pipefail

CONTAINER="${CONTAINER:-oat_mipt_robomimic_askhabaliev_gs}"
PAPER_SEED="${PAPER_SEED:-10000}"
N_EXP="${N_EXP:-5}"

# suite|gpu
JOBS=(
  "can|0"
  "coffee-pull|1"
  "stick-pull|0"
  "disassemble|1"
  "box-close|0"
)

launch_one() {
  local suite="$1" gpu="$2"
  local name="paper_s${PAPER_SEED}_${suite}"
  tmux kill-session -t "${name}" 2>/dev/null || true
  tmux new -s "${name}" -d \
    "docker exec ${CONTAINER} bash -lc '
      cd /workspace/oat &&
      TEST_START_SEED=${PAPER_SEED} N_EXP=${N_EXP} SKIP_AWR=1 FORCE_RERUN=1 GPU=${gpu} SUITE=${suite} \
        bash scripts/cluster_matched_triplet.sh
    ' 2>&1 | tee /tmp/${name}.log; echo EXIT:\$?; sleep 7200"
  echo "STARTED ${name} (SUITE=${suite} GPU=${gpu})"
}

echo "=== PAPER WAVE seed=${PAPER_SEED} n_exp=${N_EXP} SKIP_AWR=1 ==="
for row in "${JOBS[@]}"; do
  IFS='|' read -r suite gpu <<<"${row}"
  launch_one "${suite}" "${gpu}"
done

sleep 2
tmux ls | grep "paper_s${PAPER_SEED}_" || true
echo "Artifacts root: output/eval/matched_s${PAPER_SEED}/<suite>/"
echo "Logs: logs/matched_s${PAPER_SEED}_<suite>_gpu*.log"
