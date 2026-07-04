#!/bin/bash
#SBATCH --job-name=rmm_pol_lift
#SBATCH --output=slurm/logs/rmm_pol_lift_%j.out
#SBATCH --error=slurm/logs/rmm_pol_lift_%j.err
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# Phase D: train OAT policy on RoboMimic Lift with sim eval + SR checkpointing.
#
#   export TOKENIZER_CKPT=output/.../ep-xxxx_mse-0.002.ckpt
#   sbatch slurm/robomimic/train_policy_lift.slurm

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p slurm/logs
OAT_DIR="${OAT_DIR:-$(pwd)}"
[[ -d "${OAT_DIR}/scripts" ]] || OAT_DIR="$(pwd)/oat"
cd "${OAT_DIR}"

NUM_DEMO="${NUM_DEMO:-200}"
NUM_MACHINES="${NUM_MACHINES:-1}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
TOKENIZER_CKPT="${TOKENIZER_CKPT:?Set TOKENIZER_CKPT to stage-1 tokenizer .ckpt}"

HYDRA_FULL_ERROR=1 MUJOCO_GL=egl uv run accelerate launch \
  --num_machines "${NUM_MACHINES}" \
  --multi_gpu \
  --num_processes "${NUM_PROCESSES}" \
  scripts/run_workspace.py \
  --config-name=train_oatpolicy \
  task/policy=robomimic/lift \
  task.policy.lazy_eval=false \
  policy.action_tokenizer.checkpoint="${TOKENIZER_CKPT}" \
  training.num_demo="${NUM_DEMO}" \
  checkpoint.topk.k=3 \
  logging.mode=disabled
