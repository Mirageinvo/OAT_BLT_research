#!/bin/bash
#SBATCH --job-name=mw_tok
#SBATCH --output=logs/slurm_mw_tok_%j.out
#SBATCH --error=logs/slurm_mw_tok_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

export MUJOCO_GL=egl
NUM_DEMO="${NUM_DEMO:-50}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"

bash scripts/prepare_metaworld_mt4.sh tok
