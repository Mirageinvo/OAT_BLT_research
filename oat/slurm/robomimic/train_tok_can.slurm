#!/bin/bash
#SBATCH --job-name=rmm_tok_can
#SBATCH --output=slurm/logs/rmm_tok_can_%j.out
#SBATCH --error=slurm/logs/rmm_tok_can_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# Phase C: OAT tokenizer on RoboMimic Can (mh, top-3 MSE checkpoints).
# Prereq: can_mh_image.hdf5 + can_N200.zarr
#
#   cd oat && export NUM_DEMO=200
#   sbatch slurm/robomimic/train_tok_can.slurm

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p slurm/logs
OAT_DIR="${OAT_DIR:-$(pwd)}"
[[ -d "${OAT_DIR}/scripts" ]] || OAT_DIR="$(pwd)/oat"
cd "${OAT_DIR}"

NUM_DEMO="${NUM_DEMO:-200}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"

ZARR=$(ls -1d data/robomimic/can_N*.zarr 2>/dev/null | head -1)
if [[ -z "${ZARR}" ]]; then
  echo "ERROR: no can_N*.zarr — run: bash scripts/prepare_robomimic_can.sh convert"
  exit 1
fi
echo "Using zarr: ${ZARR}  NUM_DEMO=${NUM_DEMO}"

HYDRA_FULL_ERROR=1 uv run accelerate launch \
  --num_machines 1 \
  --multi_gpu \
  --num_processes "${NUM_PROCESSES}" \
  scripts/run_workspace.py \
  --config-name=train_oattok \
  task/tokenizer=robomimic/can \
  training.num_demo="${NUM_DEMO}" \
  checkpoint.topk.k=3 \
  logging.mode=disabled
