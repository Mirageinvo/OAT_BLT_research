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
#   export POLICY_CKPT=output/.../ep-xxxx_sr-0.9xx.ckpt
#   bash scripts/prepare_robomimic_lift.sh eval_base         # phase E: baseline SR
#   bash scripts/prepare_robomimic_lift.sh eval_bon          # phase F: BoN vs baseline
#   bash scripts/prepare_robomimic_lift.sh awr_collect       # phase G: BoN-distill dataset
#   bash scripts/prepare_robomimic_lift.sh awr_train         # phase H: AWR fine-tune
#   bash scripts/prepare_robomimic_lift.sh eval_awr          # phase I: AWR single-sample SR
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
  ROLLOUT_EVERY="${ROLLOUT_EVERY:-100}"
  POLICY_LOG="${POLICY_LOG:-logs/train_policy_lift.log}"
  mkdir -p logs "$(dirname "${POLICY_LOG}")"
  : > "${POLICY_LOG}"
  echo "Training policy with tokenizer: ${TOKENIZER_CKPT}"
  echo "  sim-eval (SR) every ${ROLLOUT_EVERY} epochs | top-3 ckpt by mean_success_rate"
  echo "  full log -> ${POLICY_LOG} (+ hydra run dir under output/)"
  # shellcheck disable=SC2068
  run_policy() {
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
      training.rollout_every="${ROLLOUT_EVERY}" \
      checkpoint.topk.k=3 \
      checkpoint.topk.monitor_key=mean_success_rate \
      logging.mode=disabled
  }
  if [[ -t 1 ]]; then
    run_policy 2>&1 | tee -a "${POLICY_LOG}"
  else
    run_policy >> "${POLICY_LOG}" 2>&1
  fi
}

eval_base() {
  POLICY_CKPT="${POLICY_CKPT:?Set POLICY_CKPT to baseline policy .ckpt}"
  MUJOCO_GL=egl bash scripts/eval_robomimic_policy.sh "${POLICY_CKPT}" lift
}

eval_bon() {
  POLICY_CKPT="${POLICY_CKPT:?Set POLICY_CKPT to baseline policy .ckpt}"
  BON_N="${BON_N:-8}"
  MUJOCO_GL=egl bash scripts/eval_robomimic_bon.sh "${POLICY_CKPT}" lift
}

awr_collect() {
  POLICY_CKPT="${POLICY_CKPT:?Set POLICY_CKPT to baseline policy .ckpt}"
  AWR_DATASET="${AWR_DATASET:-my_datasets/awr_lift_bon.npz}"
  AWR_N_CHUNKS="${AWR_N_CHUNKS:-20000}"
  AWR_N_WORKERS="${AWR_N_WORKERS:-6}"
  echo "Collecting BoN-distill AWR dataset -> ${AWR_DATASET}"
  MUJOCO_GL=egl uv run python scripts/collect_awr_dataset.py \
    -c "${POLICY_CKPT}" \
    -o "${AWR_DATASET}" \
    --n_chunks "${AWR_N_CHUNKS}" \
    --bon_n "${BON_N:-8}" \
    --n_workers "${AWR_N_WORKERS}"
  uv run python scripts/validate_awr.py -i "${AWR_DATASET}"
}

awr_train() {
  POLICY_CKPT="${POLICY_CKPT:?Set POLICY_CKPT to baseline policy .ckpt}"
  AWR_DATASET="${AWR_DATASET:-my_datasets/awr_lift_bon.npz}"
  AWR_CKPT="${AWR_CKPT:-my_models/policy_awr_lift.ckpt}"
  AWR_EPOCHS="${AWR_EPOCHS:-100}"
  uv run python scripts/train_awr.py \
    -i "${AWR_DATASET}" \
    -c "${POLICY_CKPT}" \
    -o "${AWR_CKPT}" \
    --beta 0.5 --beta_kl 0.05 --epochs "${AWR_EPOCHS}" --ordering uniform
}

eval_awr() {
  AWR_CKPT="${AWR_CKPT:-my_models/policy_awr_lift.ckpt}"
  MUJOCO_GL=egl bash scripts/eval_robomimic_awr.sh "${AWR_CKPT}" lift
}

case "${ACTION}" in
  download) download ;;
  convert)  convert ;;
  tok|train) tok ;;   # train = legacy alias
  policy)   policy ;;
  eval_base) eval_base ;;
  eval_bon)  eval_bon ;;
  awr_collect) awr_collect ;;
  awr_train)   awr_train ;;
  eval_awr)    eval_awr ;;
  all)      download; convert; tok ;;
  *)
    echo "Usage: $0 {download|convert|tok|policy|eval_base|eval_bon|awr_collect|awr_train|eval_awr|all}"
    exit 1
    ;;
esac
