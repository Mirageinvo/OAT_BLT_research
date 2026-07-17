#!/usr/bin/env bash
# Run a space-separated SUITES queue on one GPU (sequential within GPU).
# Usage:
#   GPU=1 SUITES="can mt4 coffee-pull" bash scripts/cluster_launch_matched_queue.sh
set -euo pipefail
cd /workspace/oat
GPU="${GPU:?Set GPU=0|1}"
SUITES="${SUITES:?Set SUITES='can mt4 ...'}"
LOG="logs/matched_queue_gpu${GPU}.log"
mkdir -p logs
echo "[queue gpu=${GPU}] $(date -Iseconds) suites=${SUITES}" | tee "${LOG}"
for s in ${SUITES}; do
  echo "[queue gpu=${GPU}] >>> ${s} $(date -Iseconds)" | tee -a "${LOG}"
  SUITE="${s}" GPU="${GPU}" bash scripts/cluster_matched_triplet.sh
  echo "[queue gpu=${GPU}] <<< ${s} DONE $(date -Iseconds)" | tee -a "${LOG}"
done
echo "[queue gpu=${GPU}] ALL DONE $(date -Iseconds)" | tee -a "${LOG}"
