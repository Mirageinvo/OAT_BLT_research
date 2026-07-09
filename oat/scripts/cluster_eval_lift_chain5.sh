#!/usr/bin/env bash
# Final paper-style eval for the trained Lift policy: ONE best checkpoint,
# 5 sequential eval passes (eval seeds 0..4), each n_test=50 → 250 episodes total.
# The next seed starts automatically after the previous run finishes.
#
# Usage (on cluster):
#   tmux new -s eval_lift_chain5 -d 'bash scripts/cluster_eval_lift_chain5.sh'
set -euo pipefail
cd /workspace/oat
source .venv/bin/activate

export OAT_USE_UV_RUN=0
# shellcheck disable=SC1091
source scripts/cluster_gpu_env.sh 1
bash scripts/patch_robosuite_egl_assert.sh

CKPT="${CKPT:-output/20260706/163500_train_oatpolicy_lift_N200/checkpoints/ep-0600_sr-0.920.ckpt}"
OUT_ROOT="${OUT_ROOT:-output/eval/robomimic_lift_paper5_ep0600}"
LOG="${LOG:-logs/eval_lift_chain5.log}"
N_TEST="${N_TEST:-50}"
BASE_SEED="${BASE_SEED:-1000}"
N_PARALLEL="${N_PARALLEL:-2}"

mkdir -p logs "${OUT_ROOT}"

echo "=== Lift chain eval | 5 seeds × n_test=${N_TEST} | OAT8 | $(date -Iseconds) ===" | tee "${LOG}"
echo "ckpt=${CKPT}" | tee -a "${LOG}"
echo "out_root=${OUT_ROOT}" | tee -a "${LOG}"
echo "gpu=${CUDA_VISIBLE_DEVICES}" | tee -a "${LOG}"

SR_VALUES=()
for EVAL_SEED in 0 1 2 3 4; do
  TEST_START=$((BASE_SEED + EVAL_SEED * N_TEST))
  OUT_DIR="${OUT_ROOT}/seed_${EVAL_SEED}"
  SEED_LOG="${OUT_ROOT}/seed_${EVAL_SEED}.log"

  echo "" | tee -a "${LOG}"
  echo ">>> eval seed ${EVAL_SEED}/4 | test_start_seed=${TEST_START} | out=${OUT_DIR} | $(date -Iseconds)" | tee -a "${LOG}"

  if [[ -d "${OUT_DIR}" ]]; then
    echo "    removing existing ${OUT_DIR}" | tee -a "${LOG}"
    rm -rf "${OUT_DIR}"
  fi

  python scripts/eval_policy_sim.py \
    -c "${CKPT}" \
    -o "${OUT_DIR}" \
    -d "${OAT_DEVICE}" \
    --num_exp 1 \
    --n_test "${N_TEST}" \
    --test_start_seed "${TEST_START}" \
    --n_parallel_envs "${N_PARALLEL}" \
    --use_k_tokens 8 \
    --entropy_threshold 0 \
    2>&1 | tee -a "${LOG}" | tee "${SEED_LOG}"

  SR="$(python - <<PY
import json
with open("${OUT_DIR}/eval_log.json") as f:
    d = json.load(f)
print(d["mean_success_rate_mean"])
PY
)"
  SR_VALUES+=("${SR}")
  echo ">>> seed ${EVAL_SEED} done: SR=${SR}" | tee -a "${LOG}"
done

python - <<PY | tee -a "${LOG}"
import json, math, pathlib
root = pathlib.Path("${OUT_ROOT}")
srs = []
for seed in range(5):
    p = root / f"seed_{seed}" / "eval_log.json"
    with open(p) as f:
        srs.append(json.load(f)["mean_success_rate_mean"])
mean = sum(srs) / len(srs)
stderr = math.sqrt(sum((x - mean) ** 2 for x in srs) / (len(srs) - 1)) / math.sqrt(len(srs)) if len(srs) > 1 else 0.0
summary = {
    "checkpoint": "${CKPT}",
    "n_test_per_seed": ${N_TEST},
    "base_seed": ${BASE_SEED},
    "per_seed_sr": {str(i): srs[i] for i in range(5)},
    "mean_success_rate": mean,
    "stderr": stderr,
    "pct_mean": 100.0 * mean,
    "pct_stderr": 100.0 * stderr,
}
out = root / "summary.json"
out.write_text(json.dumps(summary, indent=2))
print("")
print("=== FINAL (5 eval seeds, 250 episodes total) ===")
print(f"per-seed SR: {srs}")
print(f"mean ± stderr: {100*mean:.1f} ± {100*stderr:.1f}%  ({mean:.4f} ± {stderr:.4f})")
print(f"written: {out}")
PY

echo "=== chain eval complete | $(date -Iseconds) ===" | tee -a "${LOG}"
