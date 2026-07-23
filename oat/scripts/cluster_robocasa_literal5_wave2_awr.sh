#!/usr/bin/env bash
# RoboCasa Wave 2 — BoN-distill AWR (ROBOCASA.md §5).
#
# Locked vs RM/MW wave2: AWR train --epochs 100 (not 30).
# Uses .venv_robocasa; literal-5 eval (not -n 5).
#
# Usage (cluster docker):
#   SUITE=close_drawer BASE_CKPT=/path/to.ckpt GPU=0 \
#     bash scripts/cluster_robocasa_literal5_wave2_awr.sh
#
# Optional: FORCE_RECOLLECT=1  SKIP_COLLECT=1  SKIP_TRAIN=1  SKIP_EVAL=1
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

SUITE="${SUITE:?set SUITE=close_drawer|coffee_press_button|turn_off_microwave|turn_off_sink_faucet}"
case "${SUITE}" in
  close_drawer|coffee_press_button|turn_off_microwave|turn_off_sink_faucet) ;;
  *) echo "ERROR: unknown SUITE=${SUITE}"; exit 2 ;;
esac

BASE_CKPT="${BASE_CKPT:?set BASE_CKPT=... (TopK @2000 lock)}"
[[ -f "${BASE_CKPT}" ]] || { echo "ERROR missing BASE_CKPT=${BASE_CKPT}"; exit 1; }

GPU="${GPU:-0}"
COLLECT_SEED="${COLLECT_SEED:-0}"
N_CHUNKS="${N_CHUNKS:-20000}"
BON_N="${BON_N:-8}"
N_WORKERS="${N_WORKERS:-4}"
# RoboCasa paper lock — do not lower for Table P.
EPOCHS="${EPOCHS:-100}"
if [[ "${EPOCHS}" != "100" ]]; then
  echo "ERROR: RoboCasa paper AWR requires EPOCHS=100 (got EPOCHS=${EPOCHS})." >&2
  echo "       Override only for debug; do not cite non-100 AWR in RESULTS." >&2
  exit 2
fi
BETA="${BETA:-0.5}"
BETA_KL="${BETA_KL:-0.05}"
SEEDS=(10000 10001 10002 10003 10004)
N_TEST="${N_TEST:-50}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/output/eval/matched_s10000/robocasa/${SUITE}}"
AWR_DS="${AWR_DS:-${ROOT}/my_datasets/awr_s10000_robocasa_${SUITE}.npz}"
AWR_CKPT="${AWR_CKPT:-${ROOT}/my_models/awr_s10000_robocasa_${SUITE}.ckpt}"
LOG="${LOG:-${OUT_ROOT}/wave2_awr_literal5.log}"
FORCE_RECOLLECT="${FORCE_RECOLLECT:-0}"
SKIP_COLLECT="${SKIP_COLLECT:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"

VENV="${VENV:-${ROOT}/.venv_robocasa}"
if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "ERROR: missing ${VENV}. Run: bash scripts/cluster_setup_robocasa_venv.sh"
  exit 1
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
PYTHON="${VENV}/bin/python"
export CUDA_VISIBLE_DEVICES="${GPU}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
export OAT_USE_UV_RUN=0
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

mkdir -p "${OUT_ROOT}" my_datasets my_models logs
{
  echo "[wave2] suite=${SUITE} base=${BASE_CKPT}"
  echo "[wave2] collect_seed=${COLLECT_SEED} bon_n=${BON_N} n_chunks=${N_CHUNKS}"
  echo "[wave2] awr_epochs=${EPOCHS} beta=${BETA} beta_kl=${BETA_KL}"
  echo "[wave2] awr_ds=${AWR_DS}"
  echo "[wave2] awr_ckpt=${AWR_CKPT}"
  echo "[wave2] eval seeds=${SEEDS[*]} (literal-5)"
  echo "[wave2] started $(date -Iseconds)"
} | tee "${LOG}"

# Require Wave1 baseline present (matched Δ needs same seeds).
for seed in "${SEEDS[@]}"; do
  b="${OUT_ROOT}/baseline_seed${seed}/eval_log.json"
  [[ -f "${b}" ]] || {
    echo "ERROR Wave1 baseline missing: ${b}" | tee -a "${LOG}"
    echo "Run cluster_robocasa_literal5_wave1.sh first." | tee -a "${LOG}"
    exit 1
  }
done

if [[ "${SKIP_COLLECT}" != "1" ]]; then
  if [[ -f "${AWR_DS}" && "${FORCE_RECOLLECT}" != "1" ]]; then
    echo "[skip] collect — ${AWR_DS} exists (FORCE_RECOLLECT=1 to redo)" | tee -a "${LOG}"
  else
    echo "[STEP1] collect BoN-distill N=${BON_N} seed=${COLLECT_SEED}" | tee -a "${LOG}"
    "${PYTHON}" scripts/collect_awr_dataset.py \
      -c "${BASE_CKPT}" \
      -o "${AWR_DS}" \
      -d "cuda:0" \
      --seed "${COLLECT_SEED}" \
      --bon_n "${BON_N}" \
      --n_chunks "${N_CHUNKS}" \
      --n_workers "${N_WORKERS}" \
      2>&1 | tee -a "${LOG}"
    [[ -f "${AWR_DS}" ]] || { echo "ERROR missing ${AWR_DS}"; exit 1; }
  fi
fi

if [[ "${SKIP_TRAIN}" != "1" ]]; then
  echo "[STEP2] train_awr epochs=${EPOCHS} (RoboCasa lock)" | tee -a "${LOG}"
  "${PYTHON}" scripts/train_awr.py \
    -i "${AWR_DS}" \
    -c "${BASE_CKPT}" \
    -o "${AWR_CKPT}" \
    --device cuda:0 \
    --beta "${BETA}" \
    --beta_kl "${BETA_KL}" \
    --epochs "${EPOCHS}" \
    --ordering uniform \
    2>&1 | tee -a "${LOG}"
  [[ -f "${AWR_CKPT}" ]] || { echo "ERROR missing ${AWR_CKPT}"; exit 1; }
fi

if [[ "${SKIP_EVAL}" != "1" ]]; then
  echo "[STEP3] literal-5 AWR eval" | tee -a "${LOG}"
  for seed in "${SEEDS[@]}"; do
    out="${OUT_ROOT}/awr_seed${seed}"
    if [[ -f "${out}/eval_log.json" && "${FORCE_RERUN:-0}" != "1" ]]; then
      echo "[skip] ${out}/eval_log.json exists" | tee -a "${LOG}"
      continue
    fi
    mkdir -p "${out}"
    echo "[run] awr seed=${seed} -> ${out}" | tee -a "${LOG}"
    "${PYTHON}" scripts/eval_policy_sim.py \
      -c "${AWR_CKPT}" \
      -o "${out}" \
      -n 1 --n_test "${N_TEST}" --test_start_seed "${seed}" \
      --use_k_tokens 8 --entropy_threshold 0 \
      --temperature 1.0 --topk 10 \
      2>&1 | tee -a "${LOG}"
    [[ -f "${out}/eval_log.json" ]] || { echo "ERROR missing ${out}/eval_log.json"; exit 1; }
  done
  "${PYTHON}" scripts/aggregate_robocasa_literal5.py --root "${OUT_ROOT}" | tee -a "${LOG}"
fi

echo "[wave2] DONE $(date -Iseconds) summary=${OUT_ROOT}/summary_literal5.json" | tee -a "${LOG}"
