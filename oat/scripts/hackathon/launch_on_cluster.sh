#!/usr/bin/env bash
# From Mac: sync hackathon scripts → cluster → launch POC in docker (2 GPU parallel).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"

CLUSTER_HOST="${CLUSTER_HOST:-askhabaliev_gs@100.98.148.137}"
SSH_KEY="${SSH_KEY:-${HOME}/.ssh/mipt_lab}"
CONTAINER="${CONTAINER:-oat_mipt_robomimic_askhabaliev_gs}"
RSYNC_SSH="ssh -i ${SSH_KEY} -o ConnectTimeout=30"

echo "=== rsync hackathon scripts -> cluster ==="
ssh -i "${SSH_KEY}" "${CLUSTER_HOST}" "mkdir -p ~/mipt_paper/oat/scripts/hackathon ~/mipt_paper/oat/hackathon_output/logs"

rsync -avz \
  -e "${RSYNC_SSH}" \
  scripts/hackathon/ \
  "${CLUSTER_HOST}:~/mipt_paper/oat/scripts/hackathon/"

# optional: sync zarr+hdf5 from Mac if SYNC_DATA=1
if [[ "${SYNC_DATA:-0}" == "1" ]]; then
  echo "=== rsync data (zarr + hdf5) — slow ==="
  rsync -avz -e "${RSYNC_SSH}" \
    data/robomimic/lift_N200.zarr \
    data/robomimic/can_N200.zarr \
    "${CLUSTER_HOST}:~/mipt_paper/oat/data/robomimic/"
  rsync -avz -e "${RSYNC_SSH}" \
    data/robomimic/hdf5_datasets/robomimic/hdf5/lift_mh_image.hdf5 \
    data/robomimic/hdf5_datasets/robomimic/hdf5/can_mh_image.hdf5 \
    "${CLUSTER_HOST}:~/mipt_paper/oat/data/robomimic/hdf5_datasets/"
fi

echo "=== launch in docker (background) ==="
ssh -i "${SSH_KEY}" "${CLUSTER_HOST}" bash -s <<REMOTE
set -euo pipefail
CONTAINER="${CONTAINER}"
mkdir -p ~/mipt_paper/oat/hackathon_output/logs
docker start "\${CONTAINER}" 2>/dev/null || true
docker exec "\${CONTAINER}" bash -lc '
  set -euo pipefail
  cd /workspace/oat
  mkdir -p hackathon_output/logs
  chmod +x scripts/hackathon/*.sh
  nohup bash scripts/hackathon/cluster_hackathon_poc.sh \
    > hackathon_output/logs/cluster_poc_master.log 2>&1 &
  echo POC_PID=\$!
'
echo "=== tail master log (10 lines) ==="
sleep 3
docker exec "\${CONTAINER}" tail -20 /workspace/oat/hackathon_output/logs/cluster_poc_master.log 2>/dev/null || true
REMOTE

echo ""
echo "Monitor:"
echo "  ssh -i ${SSH_KEY} ${CLUSTER_HOST} 'docker exec ${CONTAINER} tail -f /workspace/oat/hackathon_output/logs/train_lift.log'"
