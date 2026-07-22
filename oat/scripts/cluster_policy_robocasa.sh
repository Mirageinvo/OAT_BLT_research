#!/usr/bin/env bash
# RoboCasa OAT policy train (close_drawer | coffee_press_button).
# Uses isolated .venv_robocasa (robosuite 1.5) — NEVER shared .venv.
#
#   TASK=coffee_press_button TOKENIZER_CKPT=... GPU=auto bash scripts/cluster_policy_robocasa.sh
#   TASK=close_drawer TOKENIZER_CKPT=... MEM_THRESH_MIB=8000 bash scripts/cluster_policy_robocasa.sh
#
# GPU=auto (default): wait until some GPU has used mem < MEM_THRESH_MIB, then train.
# Never kills existing tmux / GPU jobs.
#
# Protocol: oat/ROBOCASA.md — after G0b PASS; test_start_seed=2000; TopK SR=3.
set -euo pipefail
cd /workspace/oat
mkdir -p logs

VENV="${VENV:-/workspace/oat/.venv_robocasa}"
if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "ERROR: missing ${VENV}. Run: bash scripts/cluster_setup_robocasa_venv.sh"
  exit 1
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
export OAT_USE_UV_RUN=0
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"

TASK="${TASK:?Set TASK=close_drawer|coffee_press_button}"
case "${TASK}" in
  close_drawer|coffee_press_button) ;;
  *) echo "ERROR: unknown TASK=${TASK}"; exit 2 ;;
esac

SEED="${SEED:-0}"
NUM_DEMO="${NUM_DEMO:-200}"
ROLLOUT_EVERY="${ROLLOUT_EVERY:-100}"
N_TEST="${N_TEST:-50}"
N_PARALLEL_ENVS="${N_PARALLEL_ENVS:-2}"
NUM_WORKERS="${NUM_WORKERS:-8}"
TEST_START_SEED="${TEST_START_SEED:-2000}"
MEM_THRESH_MIB="${MEM_THRESH_MIB:-8000}"
GPU="${GPU:-auto}"

ZARR="data/robocasa/${TASK}_N${NUM_DEMO}.zarr"
LOG="${POLICY_LOG:-logs/train_policy_robocasa_${TASK}_s${SEED}.log}"

# Default tok ckpts (cluster paths from plan).
if [[ -z "${TOKENIZER_CKPT:-}" ]]; then
  case "${TASK}" in
    close_drawer)
      TOKENIZER_CKPT="output/20260720/005709_train_oattok_close_drawer_N200/checkpoints/ep-1800_mse-0.002.ckpt"
      ;;
    coffee_press_button)
      TOKENIZER_CKPT="output/20260720/041753_train_oattok_coffee_press_button_N200/checkpoints/ep-1940_mse-0.003.ckpt"
      ;;
  esac
fi

if [[ ! -f "${TOKENIZER_CKPT}" ]]; then
  echo "ERROR: tokenizer missing: ${TOKENIZER_CKPT}" | tee "${LOG}"
  exit 1
fi
if [[ ! -d "${ZARR}" ]]; then
  echo "ERROR: G0 zarr missing: ${ZARR}" | tee "${LOG}"
  exit 1
fi

# Optional G0b gate (skip with SKIP_G0B=1).
if [[ "${SKIP_G0B:-0}" != "1" ]]; then
  PARITY_LOG="logs/robocasa_success_parity_${TASK}.log"
  if [[ ! -f "${PARITY_LOG}" ]] || ! grep -q "^=== PASS" "${PARITY_LOG}"; then
    echo "ERROR: G0b not PASS (${PARITY_LOG}). Run robocasa_success_parity.py first, or SKIP_G0B=1." | tee "${LOG}"
    exit 1
  fi
fi

pick_gpu() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', ' -v t="${MEM_THRESH_MIB}" '$2+0 < t {print $1; exit}'
}

if [[ "${GPU}" == "auto" ]]; then
  echo "[wait] RoboCasa policy ${TASK} queued $(date -Iseconds) thresh=${MEM_THRESH_MIB}MiB" | tee "${LOG}"
  while true; do
    FREE="$(pick_gpu || true)"
    if [[ -n "${FREE}" ]]; then
      GPU="${FREE}"
      echo "[wait] GPU ${GPU} free enough ($(date -Iseconds))" | tee -a "${LOG}"
      break
    fi
    echo "[wait] GPUs busy; sleep 120 ($(date -Iseconds))" | tee -a "${LOG}"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tee -a "${LOG}" || true
    sleep 120
  done
fi

# shellcheck disable=SC1091
source scripts/cluster_gpu_env.sh "${GPU}"

# RoboCasa venv has its own robosuite 1.5 — patch EGL assert there (shared .venv patch does not apply).
VENV="${VENV}" bash scripts/patch_robosuite_egl_assert.sh || true

echo "=== robocasa policy | task=${TASK} | seed=${SEED} | gpu=${GPU} | test_start_seed=${TEST_START_SEED} ===" | tee -a "${LOG}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID} OAT_DEVICE=${OAT_DEVICE}" | tee -a "${LOG}"
echo "  tok=${TOKENIZER_CKPT}" | tee -a "${LOG}"
echo "  zarr=${ZARR}" | tee -a "${LOG}"
echo "  venv=${VENV}" | tee -a "${LOG}"
echo "  started $(date -Iseconds)" | tee -a "${LOG}"

if [[ -f scripts/cluster_ensure_v100_torch.sh ]]; then
  bash scripts/cluster_ensure_v100_torch.sh || true
fi

HYDRA_FULL_ERROR=1 accelerate launch \
  --num_machines 1 \
  --num_processes 1 \
  scripts/run_workspace.py \
  --config-name=train_oatpolicy \
  "task/policy=robocasa/${TASK}" \
  task.policy.lazy_eval=false \
  "task.policy.env_runner.n_test=${N_TEST}" \
  "task.policy.env_runner.n_parallel_envs=${N_PARALLEL_ENVS}" \
  "task.policy.env_runner.test_start_seed=${TEST_START_SEED}" \
  "policy.action_tokenizer.checkpoint=${TOKENIZER_CKPT}" \
  training.num_demo="${NUM_DEMO}" \
  training.rollout_every="${ROLLOUT_EVERY}" \
  training.seed="${SEED}" \
  seed="${SEED}" \
  dataloader.num_workers="${NUM_WORKERS}" \
  val_dataloader.num_workers="${NUM_WORKERS}" \
  checkpoint.topk.k=3 \
  checkpoint.topk.monitor_key=mean_success_rate \
  logging.mode=disabled \
  "$@" \
  2>&1 | tee -a "${LOG}"

echo "=== robocasa policy finished $(date -Iseconds) ===" | tee -a "${LOG}"
echo "NEXT: lock BASE_CKPT = TopK by mean_success_rate (train eval seed 2000)" | tee -a "${LOG}"
