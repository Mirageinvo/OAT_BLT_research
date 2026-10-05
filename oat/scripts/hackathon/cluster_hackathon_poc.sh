#!/usr/bin/env bash
# Run INSIDE cluster docker (/workspace/oat). Parallel lift+can train, then eval+demo.
set -euo pipefail
cd /workspace/oat

source .venv/bin/activate
export OAT_USE_UV_RUN=0
export MUJOCO_GL=egl
export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
# hf download can bump hub>1.0 and break accelerate/transformers
pip install -q "huggingface-hub>=0.34.0,<1.0" 2>/dev/null || true
bash scripts/cluster_ensure_v100_torch.sh 2>/dev/null || true

STAGE="${STAGE:-hackathon_assets}"
OUT="${OUT:-hackathon_output}"
LOG_DIR="${OUT}/logs"
POL_DIR="${OUT}/policies/poc"
PROG="${OUT}/progress"
TOTAL_SEC="${TOTAL_SEC:-9000}"
N_TEST="${N_TEST:-10}"
BATCH="${BATCH:-48}"
mkdir -p "${LOG_DIR}" "${POL_DIR}" "${PROG}"

# flat HDF5 symlinks (rsync may land nested or flat)
mkdir -p data/robomimic/hdf5_datasets
for h5 in lift_mh_image can_mh_image; do
  dest="data/robomimic/hdf5_datasets/${h5}.hdf5"
  if [[ ! -f "${dest}" ]]; then
    for cand in \
      "data/robomimic/hdf5_datasets/${h5}.hdf5" \
      "data/robomimic/hdf5_datasets/robomimic/hdf5/${h5}.hdf5"; do
      if [[ -f "${cand}" ]]; then
        ln -sf "$(basename "$(dirname "${cand}")")/$(basename "${cand}")" "${dest}" 2>/dev/null || \
          ln -sf "robomimic/hdf5/${h5}.hdf5" "${dest}" 2>/dev/null || \
          cp -l "${cand}" "${dest}" 2>/dev/null || cp "${cand}" "${dest}"
        break
      fi
    done
  fi
done

need_dl=0
for f in \
  hackathon_assets/tokenizers/lift.ckpt \
  hackathon_assets/tokenizers/can.ckpt \
  data/robomimic/lift_N200.zarr \
  data/robomimic/can_N200.zarr \
  data/robomimic/hdf5_datasets/lift_mh_image.hdf5 \
  data/robomimic/hdf5_datasets/can_mh_image.hdf5; do
  [[ -e "${f}" ]] || need_dl=1
done

if [[ "${need_dl}" == "1" ]]; then
  echo "=== HF assets (missing locally) ==="
  bash scripts/hackathon/00_download_assets.sh 2>&1 | tee "${LOG_DIR}/download.log"
fi

for f in \
  hackathon_assets/tokenizers/lift.ckpt \
  data/robomimic/lift_N200.zarr \
  data/robomimic/hdf5_datasets/lift_mh_image.hdf5; do
  [[ -e "${f}" ]] || { echo "FATAL missing ${f}"; exit 1; }
done
echo "Assets OK"

pick_run_dir() {
  local task="$1"
  ls -td output/*/*train_oatpolicy_${task}_* 2>/dev/null | head -1 || true
}

pick_latest_ckpt() {
  local run_dir="$1"
  local ckpt
  ckpt="$(ls -t "${run_dir}"/checkpoints/ep-*.ckpt 2>/dev/null | head -1 || true)"
  [[ -n "${ckpt}" ]] && { echo "${ckpt}"; return; }
  [[ -f "${run_dir}/checkpoints/latest.ckpt" ]] && echo "${run_dir}/checkpoints/latest.ckpt"
}

train_task() {
  local task="$1" gpu="$2"
  TIME_LIMIT_SEC="${TOTAL_SEC}" BATCH="${BATCH}" \
    bash scripts/hackathon/train_one_timed.sh "${task}" "${gpu}" \
    2>&1 | tee "${LOG_DIR}/train_${task}.log"
}

eval_task() {
  local task="$1" gpu="$2" ckpt="$3"
  OUT="${OUT}" GPU="${gpu}" N_TEST="${N_TEST}" \
    bash scripts/hackathon/eval_and_demo_one.sh "${task}" "${ckpt}" "${gpu}" \
    2>&1 | tee "${LOG_DIR}/eval_demo_${task}.log"
}

echo "============================================================"
echo " PARALLEL TRAIN lift@GPU0 + can@GPU1  (${TOTAL_SEC}s each)"
echo " START $(date -Iseconds)"
echo "============================================================"

train_task lift 0 &
PID_L=$!
train_task can 1 &
PID_C=$!
wait "${PID_L}" || true
wait "${PID_C}" || true

declare -A TASK_CKPT
for task in lift can; do
  run_dir="$(pick_run_dir "${task}")"
  ckpt="$(pick_latest_ckpt "${run_dir}")"
  [[ -n "${ckpt}" && -f "${ckpt}" ]] || { echo "FATAL: no ckpt ${task}"; exit 1; }
  dest="${POL_DIR}/${task}_latest.ckpt"
  cp "${ckpt}" "${dest}"
  TASK_CKPT["${task}"]="${dest}"
  echo "${task} -> ${dest}"
  if [[ -f "${run_dir}/logs.json" ]]; then
    python scripts/hackathon/plot_training_progress.py \
      --run_dir "${run_dir}" --task "${task}" --output_dir "${PROG}" || true
  fi
done

echo "============================================================"
echo " PARALLEL EVAL + DEMO"
echo "============================================================"
eval_task lift 0 "${TASK_CKPT[lift]}" &
eval_task can 1 "${TASK_CKPT[can]}" &
wait

echo "DONE $(date -Iseconds)"
