#!/usr/bin/env bash
# Download RoboMimic image datasets (paper: multi-human / mh, 200 demos per task).
# Symlinks/copies HDF5 into data/robomimic/hdf5_datasets/ for convert + env metadata.
#
# Usage:
#   cd oat
#   bash scripts/download_robomimic_datasets.sh lift
#   bash scripts/download_robomimic_datasets.sh lift can square

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

TASKS=("$@")
if [[ ${#TASKS[@]} -eq 0 ]]; then
  TASKS=(lift)
fi

DEST="data/robomimic/hdf5_datasets"
mkdir -p "${DEST}"

link_or_copy_hdf5() {
  local src="$1"
  local base
  base="$(basename "${src}")"
  if [[ -e "${DEST}/${base}" ]]; then
    return 0
  fi
  if ln -sf "$(cd "$(dirname "${src}")" && pwd)/$(basename "${src}")" "${DEST}/${base}" 2>/dev/null; then
    echo "  linked ${base}"
  else
    cp -f "${src}" "${DEST}/${base}"
    echo "  copied ${base}"
  fi
}

for task in "${TASKS[@]}"; do
  echo "=== Downloading RoboMimic ${task} (mh, image) ==="
  uv run python -m robomimic.scripts.download_datasets \
    --tasks "${task}" \
    --dataset_types mh \
    --hdf5_types image
done

echo "=== Linking HDF5 into ${DEST} ==="
FOUND=0
for pattern in \
  "${DEST}"/*.hdf5 \
  "${HOME}/robomimic/datasets"/*.hdf5 \
  "${HOME}/datasets"/*.hdf5 \
  "${ROOT}/datasets"/*.hdf5; do
  for f in ${pattern}; do
    [[ -f "${f}" ]] || continue
    link_or_copy_hdf5 "${f}"
    FOUND=1
  done
done

if [[ "${FOUND}" -eq 0 ]]; then
  echo "WARNING: no HDF5 found after download. Check robomimic download path and copy manually to ${DEST}/"
  exit 1
fi

echo "HDF5 in ${DEST}:"
ls -la "${DEST}"/*.hdf5
