#!/usr/bin/env bash
# MetaWorld MT4 pipeline (paper Appendix A): data → tok → policy → BoN → AWR.
#
# Tasks: box-close, coffee-pull, disassemble, stick-pull (mt4 suite).
# Paper: 50 expert demos per task, Da=4, eval 250 rollouts (5×50), OAT8 SR ≈ 24.4%.
#
# Usage:
#   cd oat
#   bash scripts/prepare_metaworld_mt4.sh setup_mujoco   # once per machine
#   bash scripts/prepare_metaworld_mt4.sh gen_data         # expert demos → zarr
#   bash scripts/prepare_metaworld_mt4.sh tok              # tokenizer
#   export TOKENIZER_CKPT=output/.../ep-xxxx_mse-0.00x.ckpt
#   bash scripts/prepare_metaworld_mt4.sh policy           # policy + sim eval
#   export POLICY_CKPT=output/.../ep-xxxx_sr-0.2xx.ckpt
#   bash scripts/prepare_metaworld_mt4.sh eval_base
#   bash scripts/prepare_metaworld_mt4.sh eval_bon
#   bash scripts/prepare_metaworld_mt4.sh awr_collect
#   bash scripts/prepare_metaworld_mt4.sh awr_train
#   bash scripts/prepare_metaworld_mt4.sh eval_awr
#
# Cluster: slurm/metaworld/{gen_data,train_tok,train_policy}.slurm

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

ACTION="${1:-all}"
NUM_DEMO="${NUM_DEMO:-50}"
NUM_MACHINES="${NUM_MACHINES:-1}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
MW_DEVICE="${MW_DEVICE:-cuda:0}"

run_with_project_launcher() {
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
    echo "ERROR: neither 'uv' nor 'accelerate' is available."
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

_mujoco_env() {
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
  if [[ -d "${HOME}/.mujoco/mujoco210/bin" ]]; then
    export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
  fi
  if [[ -d /usr/lib/nvidia ]]; then
    export LD_LIBRARY_PATH="/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"
  fi
}

setup_mujoco() {
  echo "MetaWorld needs MuJoCo 2.1.0 at ~/.mujoco/mujoco210 (see METAWORLD.md)."
  if [[ ! -d "${HOME}/.mujoco/mujoco210" ]]; then
    mkdir -p "${HOME}/.mujoco"
    wget -q https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz \
      -O /tmp/mujoco210.tar.gz --no-check-certificate || true
    if [[ -f /tmp/mujoco210.tar.gz ]]; then
      tar -xzf /tmp/mujoco210.tar.gz -C "${HOME}/.mujoco"
      echo "Installed ${HOME}/.mujoco/mujoco210"
    else
      echo "Download mujoco210 manually (see METAWORLD.md)."
      exit 1
    fi
  fi
  _mujoco_env
  run_with_project_launcher python -c "
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
print('metaworld OK, envs:', len(ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE))
"
}

gen_data() {
  _mujoco_env
  mkdir -p data/metaworld logs
  echo "Generating MT4 expert data (${NUM_DEMO} demos per task; total=$((NUM_DEMO * 4)))..."
  run_with_project_launcher python scripts/gen_metaworld_data.py \
    --task_name mt4 \
    --num_episodes "${NUM_DEMO}" \
    --device "${MW_DEVICE}" \
    2>&1 | tee "logs/gen_metaworld_mt4_N${NUM_DEMO}.log"
  ls -la "data/metaworld/mt4_N${NUM_DEMO}.zarr" || ls -la data/metaworld/*.zarr
}

tok() {
  ZARR="data/metaworld/mt4_N${NUM_DEMO}.zarr"
  if [[ ! -d "${ZARR}" ]]; then
    ZARR=$(ls -1d data/metaworld/mt4_N*.zarr 2>/dev/null | head -1)
  fi
  if [[ -z "${ZARR}" || ! -d "${ZARR}" ]]; then
    echo "No mt4 zarr. Run: $0 gen_data"
    exit 1
  fi
  echo "Training tokenizer: mt4, num_demo=${NUM_DEMO}, zarr=${ZARR}"
  HYDRA_FULL_ERROR=1 run_with_project_launcher accelerate launch \
    $(accelerate_launch_args) \
    scripts/run_workspace.py \
    --config-name=train_oattok \
    task/tokenizer=metaworld/mt4 \
    training.num_demo="${NUM_DEMO}" \
    checkpoint.topk.k=3 \
    logging.mode=disabled
}

tok_resume() {
  TOK_RUN_DIR="${TOK_RUN_DIR:?Set TOK_RUN_DIR to existing hydra run}"
  HYDRA_FULL_ERROR=1 run_with_project_launcher accelerate launch \
    $(accelerate_launch_args) \
    scripts/run_workspace.py \
    --config-name=train_oattok \
    task/tokenizer=metaworld/mt4 \
    training.num_demo="${NUM_DEMO}" \
    training.resume=true \
    checkpoint.topk.k=3 \
    logging.mode=disabled \
    hydra.run.dir="${TOK_RUN_DIR}"
}

policy() {
  TOKENIZER_CKPT="${TOKENIZER_CKPT:?Set TOKENIZER_CKPT}"
  ROLLOUT_EVERY="${ROLLOUT_EVERY:-200}"
  POLICY_LOG="${POLICY_LOG:-logs/train_policy_mt4.log}"
  _mujoco_env
  mkdir -p logs
  : > "${POLICY_LOG}"
  echo "Training policy with tokenizer: ${TOKENIZER_CKPT}"
  HYDRA_FULL_ERROR=1 run_with_project_launcher accelerate launch \
    $(accelerate_launch_args) \
    scripts/run_workspace.py \
    --config-name=train_oatpolicy \
    task/policy=metaworld/mt4 \
    task.policy.lazy_eval=false \
    policy.action_tokenizer.checkpoint="${TOKENIZER_CKPT}" \
    training.num_demo="${NUM_DEMO}" \
    training.rollout_every="${ROLLOUT_EVERY}" \
    checkpoint.topk.k=3 \
    checkpoint.topk.monitor_key=mean_success_rate \
    logging.mode=disabled \
    2>&1 | tee -a "${POLICY_LOG}"
}

eval_base() {
  POLICY_CKPT="${POLICY_CKPT:?Set POLICY_CKPT}"
  MUJOCO_GL=egl bash scripts/eval_metaworld_policy.sh "${POLICY_CKPT}" mt4
}

eval_bon() {
  POLICY_CKPT="${POLICY_CKPT:?Set POLICY_CKPT}"
  MUJOCO_GL=egl bash scripts/eval_metaworld_bon.sh "${POLICY_CKPT}" mt4
}

awr_collect() {
  POLICY_CKPT="${POLICY_CKPT:?Set POLICY_CKPT}"
  AWR_DATASET="${AWR_DATASET:-my_datasets/awr_mt4_bon.npz}"
  _mujoco_env
  run_with_project_launcher python scripts/collect_awr_dataset.py \
    -c "${POLICY_CKPT}" \
    -o "${AWR_DATASET}" \
    --n_chunks "${AWR_N_CHUNKS:-20000}" \
    --bon_n "${BON_N:-8}" \
    --n_tasks 4 \
    --n_workers "${AWR_N_WORKERS:-6}"
  run_with_project_launcher python scripts/validate_awr.py -i "${AWR_DATASET}"
}

awr_train() {
  POLICY_CKPT="${POLICY_CKPT:?Set POLICY_CKPT}"
  AWR_DATASET="${AWR_DATASET:-my_datasets/awr_mt4_bon.npz}"
  AWR_CKPT="${AWR_CKPT:-my_models/policy_awr_mt4.ckpt}"
  run_with_project_launcher python scripts/train_awr.py \
    -i "${AWR_DATASET}" \
    -c "${POLICY_CKPT}" \
    -o "${AWR_CKPT}" \
    --beta 0.5 --beta_kl 0.05 --epochs "${AWR_EPOCHS:-100}" --ordering uniform
}

eval_awr() {
  AWR_CKPT="${AWR_CKPT:-my_models/policy_awr_mt4.ckpt}"
  MUJOCO_GL=egl bash scripts/eval_metaworld_awr.sh "${AWR_CKPT}" mt4
}

case "${ACTION}" in
  setup_mujoco) setup_mujoco ;;
  gen_data)     gen_data ;;
  tok)          tok ;;
  tok_resume)   tok_resume ;;
  policy)       policy ;;
  eval_base)    eval_base ;;
  eval_bon)     eval_bon ;;
  awr_collect)  awr_collect ;;
  awr_train)    awr_train ;;
  eval_awr)     eval_awr ;;
  all)          setup_mujoco; gen_data; tok ;;
  *)
    echo "Usage: $0 {setup_mujoco|gen_data|tok|tok_resume|policy|eval_base|eval_bon|awr_collect|awr_train|eval_awr|all}"
    exit 1
    ;;
esac
