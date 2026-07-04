#!/bin/bash
#SBATCH --job-name=rmm_convert
#SBATCH --output=slurm/logs/rmm_convert_%j.out
#SBATCH --error=slurm/logs/rmm_convert_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p slurm/logs
# Repo root: adjust if you submit from oat/ or project root
OAT_DIR="${OAT_DIR:-$(pwd)}"
[[ -d "${OAT_DIR}/scripts" ]] || OAT_DIR="$(pwd)/oat"
cd "${OAT_DIR}"

export HDF5_DIR="${HDF5_DIR:-data/robomimic/hdf5_datasets}"

echo "=== RoboMimic HDF5 -> Zarr (prefer mh image HDF5) ==="
echo "Host: $(hostname)  OAT_DIR=${OAT_DIR}"
ls -la "${HDF5_DIR}"/*.hdf5

uv run python scripts/convert_robomimic_dataset.py \
  --root_dir data/robomimic \
  --hdf5_dir_name hdf5_datasets \
  --compression_level 5 \
  --chunk_size 1024

ls -la data/robomimic/*.zarr
