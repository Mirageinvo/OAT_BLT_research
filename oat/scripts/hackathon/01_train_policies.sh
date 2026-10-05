#!/usr/bin/env bash
# Train policy FROM SCRATCH (frozen tokenizer). Do NOT pass old policy ckpts.
set -euo pipefail
cd "$(dirname "$0")/../.."

export MUJOCO_GL="${MUJOCO_GL:-egl}"
STAGE="${STAGE:-hackathon_assets}"
OUT_ROOT="${OUT_ROOT:-hackathon_output/policies}"

NUM_DEMO="${NUM_DEMO:-200}"
HACK_EPOCHS="${HACK_EPOCHS:-601}"
ROLLOUT_EVERY="${HACK_ROLLOUT_EVERY:-100}"
ROLLOUT_START="${HACK_ROLLOUT_START:-100}"
BATCH="${BATCH:-64}"
GPUS="${GPUS:-1}"

train_one() {
  local task="$1"
  local tok="${STAGE}/tokenizers/${task}.ckpt"
  if [[ ! -f "${tok}" ]]; then
    echo "Missing tokenizer ${tok}. Run 00_download_assets.sh first."
    exit 1
  fi
  echo "========== TRAIN policy ${task} (tokenizer=${tok}) =========="
  HYDRA_FULL_ERROR=1 uv run accelerate launch \
    --num_processes "${GPUS}" \
    scripts/run_workspace.py \
    --config-name=train_oatpolicy \
    "task/policy=robomimic/${task}" \
    "policy.action_tokenizer.checkpoint=${tok}" \
    "training.num_demo=${NUM_DEMO}" \
    "training.num_epochs=${HACK_EPOCHS}" \
    "training.rollout_every=${ROLLOUT_EVERY}" \
    "training.rollout_start_epoch=${ROLLOUT_START}" \
    "training.checkpoint_every=50" \
    "dataloader.batch_size=${BATCH}" \
    "val_dataloader.batch_size=${BATCH}" \
    "checkpoint.topk.k=1" \
    "logging.mode=disabled" \
    "task.policy.lazy_eval=false"

  # symlink best/latest ckpt into hackathon_output
  local run_dir
  run_dir="$(ls -td output/*/*train_oatpolicy*${task}* 2>/dev/null | head -1)"
  if [[ -z "${run_dir}" ]]; then
    echo "Could not find output dir for ${task}"
    exit 1
  fi
  mkdir -p "${OUT_ROOT}/${task}"
  best="$(ls -t "${run_dir}"/checkpoints/ep-*_sr-*.ckpt 2>/dev/null | head -1 || true)"
  if [[ -z "${best}" ]]; then
    best="${run_dir}/checkpoints/latest.ckpt"
  fi
  cp "${best}" "${OUT_ROOT}/${task}/$(basename "${best}")"
  echo "Saved ${OUT_ROOT}/${task}/$(basename "${best}")"

  # training progress for slides (loss + sim SR curve)
  PROG="${PROG:-hackathon_output/progress}"
  uv run python scripts/hackathon/plot_training_progress.py \
    --run_dir "${run_dir}" \
    --task "${task}" \
    --output_dir "${PROG}"
  echo "Progress plot -> ${PROG}/${task}_training_curve.png"
}

for task in lift can square; do
  train_one "${task}"
done

bash scripts/hackathon/03_plot_progress.sh

echo "All three policies trained. Progress -> hackathon_output/progress/"
echo "Next: bash scripts/hackathon/02_collect_success_gifs.sh"
