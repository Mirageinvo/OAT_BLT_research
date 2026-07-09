#!/usr/bin/env bash
# RoboMimic Square — phase C tokenizer (paper-aligned: mh image HDF5, top-k MSE ckpts).
#
#   cd oat
#   bash scripts/prepare_robomimic_square.sh convert
#   bash scripts/prepare_robomimic_square.sh tok
#   TOK_RUN_DIR=output/... bash scripts/prepare_robomimic_square.sh tok_resume
#
# Prereq: data/robomimic/hdf5_datasets/square_mh_image.hdf5 (from extract pipeline).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

ACTION="${1:-all}"
HDF5="${HDF5:-data/robomimic/hdf5_datasets/square_mh_image.hdf5}"
NUM_DEMO="${NUM_DEMO:-200}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
OAT_USE_UV_RUN="${OAT_USE_UV_RUN:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

run_with_project_launcher() {
  if [[ "${OAT_USE_UV_RUN}" == "1" ]]; then
    uv run "$@"
  else
    "$@"
  fi
}

accelerate_launch_args() {
  if [[ "${NUM_PROCESSES}" -gt 1 ]]; then
    printf '%s\n' --num_machines 1 --multi_gpu --num_processes "${NUM_PROCESSES}"
  else
    printf '%s\n' --num_machines 1 --num_processes 1
  fi
}

convert() {
  if [[ ! -f "${HDF5}" ]]; then
    echo "Missing ${HDF5} — run square image extract first."
    exit 1
  fi
  echo "Converting Square HDF5 -> zarr (${HDF5}, n=${NUM_DEMO})..."
  run_with_project_launcher python scripts/convert_robomimic_dataset.py \
    --root_dir data/robomimic \
    --hdf5_dir_name hdf5_datasets \
    --hdf5 square_mh_image.hdf5 \
    --skip-existing \
    --compression_level 5 \
    --chunk_size 1024 \
    -n "${NUM_DEMO}"
  ls -la data/robomimic/square_N*.zarr
}

tok_resume() {
  TOK_RUN_DIR="${TOK_RUN_DIR:?Set TOK_RUN_DIR=output/.../train_oattok_square_N200}"
  [[ -d "${TOK_RUN_DIR}/checkpoints" ]] || { echo "No checkpoints in ${TOK_RUN_DIR}"; exit 1; }
  echo "Resume Square tokenizer in ${TOK_RUN_DIR} (target test_reconst_mse ~ 0.002)"
  HYDRA_FULL_ERROR=1 run_with_project_launcher accelerate launch \
    $(accelerate_launch_args) \
    scripts/run_workspace.py \
    --config-name=train_oattok \
    task/tokenizer=robomimic/square \
    training.num_demo="${NUM_DEMO}" \
    training.resume=true \
    checkpoint.topk.k=3 \
    logging.mode=disabled \
    hydra.run.dir="${TOK_RUN_DIR}"
}

tok() {
  ZARR=$(ls -1d data/robomimic/square_N*.zarr 2>/dev/null | head -1)
  if [[ -z "${ZARR}" ]]; then
    echo "No square zarr. Run: $0 convert"
    exit 1
  fi
  if [[ "${NUM_DEMO}" == "200" ]]; then
    case "${ZARR}" in
      *_N*.zarr)
        NUM_DEMO="${ZARR##*_N}"
        NUM_DEMO="${NUM_DEMO%.zarr}"
        ;;
    esac
  fi
  echo "Training Square tokenizer: num_demo=${NUM_DEMO}, zarr=${ZARR}, topk=3"
  HYDRA_FULL_ERROR=1 run_with_project_launcher accelerate launch \
    $(accelerate_launch_args) \
    scripts/run_workspace.py \
    --config-name=train_oattok \
    task/tokenizer=robomimic/square \
    training.num_demo="${NUM_DEMO}" \
    checkpoint.topk.k=3 \
    logging.mode=disabled
}

case "${ACTION}" in
  convert) convert ;;
  tok|train) tok ;;
  tok_resume) tok_resume ;;
  all) convert; tok ;;
  *)
    echo "Usage: $0 {convert|tok|tok_resume|all}"
    exit 1
    ;;
esac
