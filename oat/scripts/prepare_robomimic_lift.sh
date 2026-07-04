#!/usr/bin/env bash
# RoboMimic Lift pipeline (paper-aligned: mh data, top-k checkpoints, sim eval).
#
# Usage:
#   cd oat
#   bash scripts/prepare_robomimic_lift.sh download          # mh HDF5
#   bash scripts/prepare_robomimic_lift.sh convert           # HDF5 -> zarr
#   bash scripts/prepare_robomimic_lift.sh tok               # phase C: tokenizer
#   export TOKENIZER_CKPT=output/.../ep-xxxx_mse-0.002.ckpt
#   bash scripts/prepare_robomimic_lift.sh policy            # phase D: policy
#   bash scripts/prepare_robomimic_lift.sh all               # download+convert+tok
#
# Cluster (SLURM): slurm/robomimic/{convert_lift,train_tok_lift,train_policy_lift}.slurm

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

ACTION="${1:-all}"
HDF5_DIR="${HDF5_DIR:-data/robomimic/hdf5_datasets}"
NUM_DEMO="${NUM_DEMO:-200}"
NUM_MACHINES="${NUM_MACHINES:-1}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"

download() {
  bash scripts/download_robomimic_datasets.sh lift
}

convert() {
  if ! ls "${HDF5_DIR}"/*lift*.hdf5 &>/dev/null; then
    echo "No Lift HDF5 in ${HDF5_DIR}"
    echo "Run: $0 download"
    exit 1
  fi
  if ! ls "${HDF5_DIR}"/*mh*.hdf5 &>/dev/null; then
    echo "WARNING: no *mh* HDF5 found — paper uses multi-human (mh) demos."
  fi
  echo "Converting HDF5 -> Zarr (root=${ROOT}, hdf5_dir=${HDF5_DIR})..."
  uv run python scripts/convert_robomimic_dataset.py \
    --root_dir data/robomimic \
    --hdf5_dir_name hdf5_datasets \
    --compression_level 5 \
    --chunk_size 1024 \
    -n "${NUM_DEMO}"
  echo "Expected zarr: data/robomimic/lift_N*.zarr"
  ls -la data/robomimic/*.zarr 2>/dev/null || true
}

tok() {
  ZARR=$(ls -1 data/robomimic/lift_N*.zarr 2>/dev/null | head -1)
  if [[ -z "${ZARR}" ]]; then
    echo "No lift zarr found. Run: $0 convert"
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
  echo "Training tokenizer: task=lift, num_demo=${NUM_DEMO}, zarr=${ZARR}"
  HYDRA_FULL_ERROR=1 uv run accelerate launch \
    --num_machines "${NUM_MACHINES}" \
    --multi_gpu \
    --num_processes "${NUM_PROCESSES}" \
    scripts/run_workspace.py \
    --config-name=train_oattok \
    task/tokenizer=robomimic/lift \
    training.num_demo="${NUM_DEMO}" \
    checkpoint.topk.k=3 \
    logging.mode=disabled
}

policy() {
  TOKENIZER_CKPT="${TOKENIZER_CKPT:?Set TOKENIZER_CKPT to stage-1 tokenizer .ckpt}"
  echo "Training policy with tokenizer: ${TOKENIZER_CKPT}"
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
}

case "${ACTION}" in
  download) download ;;
  convert)  convert ;;
  tok|train) tok ;;   # train = legacy alias
  policy)   policy ;;
  all)      download; convert; tok ;;
  *)
    echo "Usage: $0 {download|convert|tok|policy|all}"
    exit 1
    ;;
esac
