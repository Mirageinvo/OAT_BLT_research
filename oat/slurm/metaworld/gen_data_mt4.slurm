#!/bin/bash
#SBATCH --job-name=mw_gen
#SBATCH --output=logs/slurm_mw_gen_%j.out
#SBATCH --error=logs/slurm_mw_gen_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs data/metaworld

export MUJOCO_GL=egl
if [[ -d "${HOME}/.mujoco/mujoco210/bin" ]]; then
  export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
fi

NUM_DEMO="${NUM_DEMO:-50}"
MW_DEVICE="${MW_DEVICE:-cuda:0}"

bash scripts/prepare_metaworld_mt4.sh setup_mujoco
bash scripts/prepare_metaworld_mt4.sh gen_data
