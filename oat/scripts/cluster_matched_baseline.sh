#!/usr/bin/env bash
# Lab matched BASELINE only (Table B) — NOT the paper path.
# Paper = scripts/cluster_matched_triplet.sh with TEST_START_SEED=10000, -n 5.
#
# Defaults below match paper TopK locks where possible; protocol here stays
# seed=1000 / n_exp=3 (selection-pool lab). Prefer triplet for anything cited.
#
# Usage: SUITE=can GPU=1 bash scripts/cluster_matched_baseline.sh
set -euo pipefail
cd /workspace/oat
source .venv/bin/activate
export MUJOCO_GL=egl
export OAT_USE_UV_RUN=0

SUITE="${SUITE:?Set SUITE}"
GPU="${GPU:-0}"
# shellcheck disable=SC1091
source scripts/cluster_gpu_env.sh "${GPU}"
bash scripts/patch_robosuite_egl_assert.sh

N_TEST="${N_TEST:-50}"
TEST_START_SEED="${TEST_START_SEED:-1000}"
N_EXP="${N_EXP:-3}"
N_PARALLEL="${N_PARALLEL:-4}"

case "${SUITE}" in
  lift)
    # Paper primary = ep-1400; override CKPT= for run A ep-0900 if needed.
    CKPT="${CKPT:-output/20260719/144024_train_oatpolicy_lift_N200/checkpoints/ep-1400_sr-0.950.ckpt}"
    ENV_TASK=""
    ;;
  can)
    CKPT="${CKPT:-output/20260706/173343_train_oatpolicy_can_N200/checkpoints/ep-1700_sr-0.940.ckpt}"
    ENV_TASK=""
    ;;
  square)
    CKPT="${CKPT:-output/20260720/215024_train_oatpolicy_square_N200/checkpoints/ep-0700_sr-0.420.ckpt}"
    ENV_TASK=""
    ;;
  mt4)
    echo "ERROR: SUITE=mt4 removed from paper track (policy run deleted). Use MW specialists." >&2
    exit 1
    ;;
  coffee-pull)
    CKPT="${CKPT:-output/20260720/090816_train_oatpolicy_mw-coffee-pull_st_N50/checkpoints/ep-1000_sr-0.432.ckpt}"
    ENV_TASK="coffee-pull"
    ;;
  stick-pull)
    CKPT="${CKPT:-output/20260711/134439_train_oatpolicy_mw-stick-pull_st_N50/checkpoints/ep-0800_sr-0.212.ckpt}"
    ENV_TASK="stick-pull"
    ;;
  disassemble)
    CKPT="${CKPT:-output/20260711/134440_train_oatpolicy_mw-disassemble_st_N50/checkpoints/ep-1400_sr-0.700.ckpt}"
    ENV_TASK="disassemble"
    ;;
  box-close)
    CKPT="${CKPT:-output/20260711/134439_train_oatpolicy_mw-box-close_st_N50/checkpoints/ep-2000_sr-0.552.ckpt}"
    ENV_TASK="box-close"
    ;;
  *) echo "Unknown SUITE=${SUITE}" >&2; exit 1 ;;
esac

OUT="${OUT:-output/eval/matched/${SUITE}/baseline_n3}"
LOG="${LOG:-logs/matched_${SUITE}_baseline_gpu${GPU}.log}"
mkdir -p logs "$(dirname "${OUT}")"
rm -rf "${OUT}"

env_args=()
[[ -n "${ENV_TASK}" ]] && env_args+=(--env_task_name "${ENV_TASK}")

echo "=== MATCHED BASELINE ${SUITE} ===" | tee "${LOG}"
echo "ckpt=${CKPT}" | tee -a "${LOG}"
echo "out=${OUT} n_test=${N_TEST} seed=${TEST_START_SEED} -n ${N_EXP} gpu=${CUDA_VISIBLE_DEVICES}" | tee -a "${LOG}"
[[ -f "${CKPT}" ]] || { echo "ERROR missing ckpt"; exit 1; }

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl \
  python scripts/eval_policy_sim.py \
    -c "${CKPT}" \
    -o "${OUT}" \
    -d "${OAT_DEVICE}" \
    -n "${N_EXP}" \
    --n_test "${N_TEST}" \
    --test_start_seed "${TEST_START_SEED}" \
    --n_parallel_envs "${N_PARALLEL}" \
    --use_k_tokens 8 \
    --entropy_threshold 0 \
    "${env_args[@]}" \
  2>&1 | tee -a "${LOG}"

python - <<PY | tee -a "${LOG}"
import json
from pathlib import Path
p = Path("${OUT}") / "eval_log.json"
d = json.load(p.open())
m, s = float(d["mean_success_rate_mean"]), float(d.get("mean_success_rate_std", 0.0))
print(f"=== FINAL matched baseline ${SUITE}: {100*m:.1f} ± {100*s:.1f}% ===")
PY
echo "[DONE] ${SUITE} $(date -Iseconds)" | tee -a "${LOG}"
