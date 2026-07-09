#!/usr/bin/env bash
# Sync oat/ to ccmplanner (run from your Mac — needs Tailscale/VPN).
#
#   cd oat
#   bash scripts/sync_to_cluster.sh
#
# Optional:
#   CLUSTER_HOST=askhabaliev_gs@100.98.148.137
#   CLUSTER_OAT=~/mipt_paper/oat
#   SSH_KEY=~/.ssh/mipt_lab

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

CLUSTER_HOST="${CLUSTER_HOST:-askhabaliev_gs@100.98.148.137}"
CLUSTER_OAT="${CLUSTER_OAT:-~/mipt_paper/oat}"
SSH_KEY="${SSH_KEY:-${HOME}/.ssh/mipt_lab}"
RSYNC_SSH="ssh -i ${SSH_KEY} -o ConnectTimeout=20"

echo "=== rsync ${ROOT}/ -> ${CLUSTER_HOST}:${CLUSTER_OAT}/ ==="
ssh -i "${SSH_KEY}" -o ConnectTimeout=20 "${CLUSTER_HOST}" "mkdir -p ${CLUSTER_OAT}"

rsync -avz --delete \
  -e "${RSYNC_SSH}" \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude 'output' \
  --exclude 'data/robomimic' \
  --exclude 'data/metaworld' \
  --exclude 'my_datasets' \
  --exclude 'eval_out' \
  --exclude 'logs' \
  --exclude '.git' \
  "${ROOT}/" "${CLUSTER_HOST}:${CLUSTER_OAT}/"

echo "=== done. On cluster: bash scripts/cluster_ccmplanner.sh <cmd> ==="
