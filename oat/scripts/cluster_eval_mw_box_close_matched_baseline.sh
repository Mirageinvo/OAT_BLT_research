#!/usr/bin/env bash
# ICRA matched baseline eval for MetaWorld box-close (specialist).
# Same protocol as RESOLUTIONPLAN / LIBERO lead track estimator:
#   test_start_seed=1000, n_test=50, num_exp=3, OAT8
# NOT chain5 (5×50 disjoint blocks) — that is sanity/appendix only.
#
# Usage (cluster):
#   tmux new -s mwst_matched_box_close -d 'bash scripts/cluster_eval_mw_box_close_matched_baseline.sh'
set -euo pipefail
cd /workspace/oat
source .venv/bin/activate
export OAT_USE_UV_RUN=0
export MUJOCO_GL=egl

GPU="${GPU:-0}"
# shellcheck disable=SC1091
source scripts/cluster_gpu_env.sh "${GPU}"
bash scripts/patch_robosuite_egl_assert.sh

CKPT="${CKPT:-output/20260711/134439_train_oatpolicy_mw-box-close_st_N50/checkpoints/ep-2000_sr-0.552.ckpt}"
OUT="${OUT:-output/eval/matched/metaworld_box-close/baseline_n3}"
LOG="${LOG:-logs/eval_mw_box_close_matched_baseline.log}"
N_TEST="${N_TEST:-50}"
TEST_START_SEED="${TEST_START_SEED:-1000}"
N_EXP="${N_EXP:-3}"
N_PARALLEL="${N_PARALLEL:-4}"

mkdir -p logs "$(dirname "${OUT}")"
# eval_policy_sim.py prompts if OUT exists — remove so run is non-interactive
rm -rf "${OUT}"

echo "=== MetaWorld box-close MATCHED baseline ===" | tee "${LOG}"
echo "ckpt=${CKPT}" | tee -a "${LOG}"
echo "out=${OUT} n_test=${N_TEST} test_start_seed=${TEST_START_SEED} n_exp=${N_EXP}" | tee -a "${LOG}"
echo "gpu=${CUDA_VISIBLE_DEVICES} device=${OAT_DEVICE} $(date -Iseconds)" | tee -a "${LOG}"

if [[ ! -f "${CKPT}" ]]; then
  echo "ERROR: checkpoint not found: ${CKPT}" | tee -a "${LOG}"
  exit 1
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl \
  python scripts/eval_policy_sim.py \
    -c "${CKPT}" \
    -o "${OUT}" \
    -d "${OAT_DEVICE}" \
    -n "${N_EXP}" \
    --n_test "${N_TEST}" \
    --test_start_seed "${TEST_START_SEED}" \
    --n_parallel_envs "${N_PARALLEL}" \
    --env_task_name box-close \
    --use_k_tokens 8 \
    --entropy_threshold 0 \
  2>&1 | tee -a "${LOG}"

python - <<PY | tee -a "${LOG}"
import json, pathlib
p = pathlib.Path("${OUT}") / "eval_log.json"
d = json.load(p.open())
mean = float(d["mean_success_rate_mean"])
std = float(d.get("mean_success_rate_std", 0.0))
print("")
print("=== FINAL matched baseline box-close ===")
print(f"SR = {100*mean:.1f} ± {100*std:.1f}%  (n_test=${N_TEST}, seed=${TEST_START_SEED}, -n ${N_EXP})")
print(f"log: {p}")
PY

echo "=== DONE $(date -Iseconds) ===" | tee -a "${LOG}"
