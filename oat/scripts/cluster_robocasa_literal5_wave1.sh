#!/usr/bin/env bash
# RoboCasa Wave 1 — literal 5 seeds (ROBOCASA.md §4).
# Same BASE_CKPT for all seeds; separate OUT dirs; then aggregate.
#
# Usage (inside oat docker / env with MuJoCo):
#   SUITE=close_drawer BASE_CKPT=/path/to.ckpt GPU=0 \
#     bash scripts/cluster_robocasa_literal5_wave1.sh
#
# Optional: SKIP_BON=1  SKIP_BASELINE=1  FORCE_RERUN=1
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

SUITE="${SUITE:?set SUITE=close_drawer|coffee_press_button|turn_off_microwave|turn_off_sink_faucet}"
BASE_CKPT="${BASE_CKPT:?set BASE_CKPT=...}"
GPU="${GPU:-0}"
SEEDS=(10000 10001 10002 10003 10004)
N_TEST="${N_TEST:-50}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/output/eval/matched_s10000/robocasa/${SUITE}}"
SKIP_BASELINE="${SKIP_BASELINE:-0}"
SKIP_BON="${SKIP_BON:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"
VENV="${VENV:-${ROOT}/.venv_robocasa}"
export CUDA_VISIBLE_DEVICES="${GPU}"
# After CUDA_VISIBLE_DEVICES remaps, EGL only sees device 0 in the visible set.
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
export OAT_USE_UV_RUN=0

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "ERROR: missing ${VENV}. RoboCasa eval needs .venv_robocasa (robosuite 1.5)."
  exit 1
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
PYTHON="${VENV}/bin/python"

mkdir -p "${OUT_ROOT}"
LOG="${OUT_ROOT}/wave1_literal5.log"
echo "[wave1] suite=${SUITE} ckpt=${BASE_CKPT} seeds=${SEEDS[*]} venv=${VENV}" | tee "${LOG}"

run_one() {
  local method="$1" seed="$2" extra=("${@:3}")
  local out="${OUT_ROOT}/${method}_seed${seed}"
  if [[ "${FORCE_RERUN}" == "1" ]]; then
    rm -rf "${out}"
  elif [[ -f "${out}/eval_log.json" ]]; then
    echo "[skip] ${out}/eval_log.json exists" | tee -a "${LOG}"
    return 0
  fi
  # Do NOT mkdir "${out}" before eval: eval_policy_sim prompts overwrite if the dir exists.
  echo "[run] ${method} seed=${seed} -> ${out}" | tee -a "${LOG}"
  "${PYTHON}" scripts/eval_policy_sim.py \
    -c "${BASE_CKPT}" \
    -o "${out}" \
    -n 1 --n_test "${N_TEST}" --test_start_seed "${seed}" \
    --use_k_tokens 8 --entropy_threshold 0 \
    --temperature 1.0 --topk 10 \
    "${extra[@]}" \
    2>&1 | tee -a "${LOG}"
  [[ -f "${out}/eval_log.json" ]] || { echo "ERROR missing ${out}/eval_log.json" | tee -a "${LOG}"; exit 1; }
}

if [[ "${SKIP_BASELINE}" != "1" ]]; then
  for seed in "${SEEDS[@]}"; do
    run_one baseline "${seed}"
  done
fi

if [[ "${SKIP_BON}" != "1" ]]; then
  for seed in "${SEEDS[@]}"; do
    run_one bon_n8 "${seed}" --bon_free 8 --bon_signal vote
  done
fi

"${PYTHON}" scripts/aggregate_robocasa_literal5.py --root "${OUT_ROOT}" | tee -a "${LOG}"
echo "[wave1] DONE ${OUT_ROOT}/summary_literal5.json" | tee -a "${LOG}"
