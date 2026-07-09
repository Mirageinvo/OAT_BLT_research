#!/bin/bash
#SBATCH --job-name=mw_pol
#SBATCH --output=logs/slurm_mw_pol_%j.out
#SBATCH --error=logs/slurm_mw_pol_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=48:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

export MUJOCO_GL=egl
if [[ -d "${HOME}/.mujoco/mujoco210/bin" ]]; then
  export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
fi

: "${TOKENIZER_CKPT:?Set TOKENIZER_CKPT before sbatch}"
NUM_DEMO="${NUM_DEMO:-50}"
ROLLOUT_EVERY="${ROLLOUT_EVERY:-200}"

bash scripts/prepare_metaworld_mt4.sh policy
