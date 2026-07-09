#!/usr/bin/env bash
# RoboMimic Lift pipeline (paper-aligned: mh data, top-k checkpoints, sim eval).
#
# Usage:
#   cd oat
#   bash scripts/prepare_robomimic_lift.sh download          # mh HDF5
#   bash scripts/prepare_robomimic_lift.sh convert           # HDF5 -> zarr
#   bash scripts/prepare_robomimic_lift.sh tok               # phase C: tokenizer
#   TOK_RUN_DIR=output/... bash scripts/prepare_robomimic_lift.sh tok_resume  # continue to MSE~0.002
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

run_with_project_launcher() {
  # On cluster V100: `uv run` re-syncs torch 2.10+cu128 (breaks cuDNN). Prefer bare venv.
  local use_uv=1
  if [[ "${OAT_USE_UV_RUN:-auto}" == "0" ]]; then
    use_uv=0
  elif [[ "${OAT_USE_UV_RUN:-auto}" == "auto" && -n "${VIRTUAL_ENV:-}" ]]; then
  if command -v accelerate >/dev/null 2>&1 || python -c "import accelerate" >/dev/null 2>&1; then
      use_uv=0
    fi
  fi
  if [[ "${use_uv}" == "1" ]] && command -v uv >/dev/null 2>&1; then
    uv run "$@"
  elif command -v accelerate >/dev/null 2>&1; then
    "$@"
  elif python -c "import accelerate" >/dev/null 2>&1; then
    python -m accelerate.commands.launch "${@:3}"
  else
    echo "ERROR: neither 'uv' nor 'accelerate' is available in this shell."
    echo "Activate the project env first (e.g. source .venv/bin/activate)."
    exit 1
  fi
}

accelerate_launch_args() {
  local args=(--num_machines "${NUM_MACHINES}" --num_processes "${NUM_PROCESSES}")
  if [[ "${NUM_PROCESSES}" -gt 1 ]]; then
    args=(--multi_gpu "${args[@]}")
  fi
  printf '%s\n' "${args[@]}"
}

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

tok_resume() {
  TOK_RUN_DIR="${TOK_RUN_DIR:?Set TOK_RUN_DIR to existing hydra run (output/.../train_oattok_lift_N200)}"
  if [[ ! -d "${TOK_RUN_DIR}/checkpoints" ]]; then
    echo "No checkpoints in ${TOK_RUN_DIR}"
    exit 1
  fi
  echo "Resume tokenizer in ${TOK_RUN_DIR} (target test_reconst_mse ~ 0.002)"
  HYDRA_FULL_ERROR=1 run_with_project_launcher accelerate launch \
    $(accelerate_launch_args) \
    scripts/run_workspace.py \
    --config-name=train_oattok \
    task/tokenizer=robomimic/lift \
    training.num_demo="${NUM_DEMO}" \
    training.resume=true \
    checkpoint.topk.k=3 \
    logging.mode=disabled \
    hydra.run.dir="${TOK_RUN_DIR}"
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
  HYDRA_FULL_ERROR=1 run_with_project_launcher accelerate launch \
    $(accelerate_launch_args) \
    scripts/run_workspace.py \
    --config-name=train_oattok \
    task/tokenizer=robomimic/lift \
    training.num_demo="${NUM_DEMO}" \
    checkpoint.topk.k=3 \
    logging.mode=disabled
}

_mujoco_env() {
  # robomimic 0.3 + mujoco_py on ccmplanner docker
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
  if [[ -d "${HOME}/.mujoco/mujoco210/bin" ]]; then
    export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
  fi
}

_cluster_torch_env() {
  # ccmplanner V100: PyTorch 2.10+cu128 cuDNN 9.2 needs SM>=7.5
  if [[ -f scripts/cluster_ensure_v100_torch.sh ]]; then
    bash scripts/cluster_ensure_v100_torch.sh
  fi
}

_policy_speed_overrides() {
  # Optional: POLICY_PROFILE=aggressive for max throughput on 2×V100-32GB (ccmplanner).
  local profile="${POLICY_PROFILE:-default}"
  case "${profile}" in
    aggressive)
      echo "  speed profile: aggressive (batch=1280, workers=12, n_parallel_envs=8, LR×5)"
      _POLICY_SPEED_OVERRIDES=(
        dataloader.batch_size=1280
        val_dataloader.batch_size=1280
        dataloader.num_workers=12
        val_dataloader.num_workers=12
        task.policy.env_runner.n_parallel_envs=8
        optimizer.policy_lr=2.5e-4
        optimizer.obs_enc_lr=5e-5
      )
      ;;
    *)
      _POLICY_SPEED_OVERRIDES=()
      ;;
  esac
}

policy() {
  TOKENIZER_CKPT="${TOKENIZER_CKPT:?Set TOKENIZER_CKPT to stage-1 tokenizer .ckpt}"
  ROLLOUT_EVERY="${ROLLOUT_EVERY:-100}"
  POLICY_LOG="${POLICY_LOG:-logs/train_policy_lift.log}"
  _cluster_torch_env
  _mujoco_env
  _policy_speed_overrides
  mkdir -p logs "$(dirname "${POLICY_LOG}")"
  : > "${POLICY_LOG}"
  echo "Training policy with tokenizer: ${TOKENIZER_CKPT}"
  echo "  sim-eval (SR) every ${ROLLOUT_EVERY} epochs | top-3 ckpt by mean_success_rate"
  echo "  full log -> ${POLICY_LOG} (+ hydra run dir under output/)"
  # shellcheck disable=SC2068
  run_policy() {
    HYDRA_FULL_ERROR=1 run_with_project_launcher accelerate launch \
      $(accelerate_launch_args) \
      scripts/run_workspace.py \
      --config-name=train_oatpolicy \
      task/policy=robomimic/lift \
      task.policy.lazy_eval=false \
      policy.action_tokenizer.checkpoint="${TOKENIZER_CKPT}" \
      training.num_demo="${NUM_DEMO}" \
      training.rollout_every="${ROLLOUT_EVERY}" \
      checkpoint.topk.k=3 \
      checkpoint.topk.monitor_key=mean_success_rate \
      logging.mode=disabled \
      "${_POLICY_SPEED_OVERRIDES[@]}"
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
  tok_resume) tok_resume ;;
  policy)   policy ;;
  eval_base) eval_base ;;
  eval_bon)  eval_bon ;;
  awr_collect) awr_collect ;;
  awr_train)   awr_train ;;
  eval_awr)    eval_awr ;;
  all)      download; convert; tok ;;
  *)
    echo "Usage: $0 {download|convert|tok|tok_resume|policy|eval_base|eval_bon|awr_collect|awr_train|eval_awr|all}"
    exit 1
    ;;
esac
