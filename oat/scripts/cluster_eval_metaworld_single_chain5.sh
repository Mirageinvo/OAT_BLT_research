#!/usr/bin/env bash
# Paper-style single-task MetaWorld eval:
# 5 eval seeds × 50 rollouts = 250 episodes per task.
#
# Usage:
#   TASK=box-close CKPT=output/.../ep-xxxx_sr-0.xxx.ckpt \
#   bash scripts/cluster_eval_metaworld_single_chain5.sh
set -euo pipefail
cd /workspace/oat
source .venv/bin/activate

export OAT_USE_UV_RUN=0
GPU="${GPU:-0}"
# shellcheck disable=SC1091
source scripts/cluster_gpu_env.sh "${GPU}"

TASK="${TASK:?Set TASK, e.g. box-close}"
CKPT="${CKPT:?Set CKPT path}"
N_TEST="${N_TEST:-50}"
BASE_SEED="${BASE_SEED:-1000}"
N_PARALLEL="${N_PARALLEL:-4}"
OUT_ROOT="${OUT_ROOT:-output/eval/metaworld_${TASK}_paper5_$(basename "${CKPT%.ckpt}")}"
LOG="${LOG:-logs/eval_${TASK}_chain5.log}"

mkdir -p logs "${OUT_ROOT}"

echo "=== MetaWorld single-task chain5 | task=${TASK} | ckpt=${CKPT} ===" | tee "${LOG}"
echo "out_root=${OUT_ROOT} n_test=${N_TEST} base_seed=${BASE_SEED}" | tee -a "${LOG}"
echo "gpu=${CUDA_VISIBLE_DEVICES} device=${OAT_DEVICE}" | tee -a "${LOG}"

if [[ ! -f "${CKPT}" ]]; then
  echo "ERROR: checkpoint not found: ${CKPT}" | tee -a "${LOG}"
  exit 1
fi

for EVAL_SEED in 0 1 2 3 4; do
  TEST_START=$((BASE_SEED + EVAL_SEED * N_TEST))
  OUT_DIR="${OUT_ROOT}/seed_${EVAL_SEED}"
  SEED_LOG="${OUT_ROOT}/seed_${EVAL_SEED}.log"

  echo "" | tee -a "${LOG}"
  echo ">>> eval seed ${EVAL_SEED}/4 | test_start_seed=${TEST_START}" | tee -a "${LOG}"
  rm -rf "${OUT_DIR}"

  python scripts/eval_policy_sim.py \
    -c "${CKPT}" \
    -o "${OUT_DIR}" \
    -d "${OAT_DEVICE}" \
    --num_exp 1 \
    --n_test "${N_TEST}" \
    --test_start_seed "${TEST_START}" \
    --n_parallel_envs "${N_PARALLEL}" \
    --env_task_name "${TASK}" \
    --use_k_tokens 8 \
    --entropy_threshold 0 \
    2>&1 | tee -a "${LOG}" | tee "${SEED_LOG}"
done

python - <<PY | tee -a "${LOG}"
import json, math, pathlib

root = pathlib.Path("${OUT_ROOT}")
task = "${TASK}"
srs = []
for i in range(5):
    p = root / f"seed_{i}" / "eval_log.json"
    with open(p) as f:
        d = json.load(f)
    srs.append(float(d["mean_success_rate_mean"]))

mean_sr = sum(srs) / len(srs)
stderr = (
    math.sqrt(sum((x - mean_sr) ** 2 for x in srs) / (len(srs) - 1)) / math.sqrt(len(srs))
    if len(srs) > 1 else 0.0
)

summary = {
    "task": task,
    "checkpoint": "${CKPT}",
    "n_test_per_seed": ${N_TEST},
    "base_seed": ${BASE_SEED},
    "per_seed_sr": {str(i): srs[i] for i in range(5)},
    "mean_success_rate": mean_sr,
    "stderr": stderr,
    "pct_mean": 100 * mean_sr,
    "pct_stderr": 100 * stderr,
}
out = root / "summary.json"
out.write_text(json.dumps(summary, indent=2))

print("")
print("=== FINAL MetaWorld single-task chain5 ===")
print(f"task: {task}")
print(f"per-seed SR: {srs}")
print(f"mean ± stderr: {100*mean_sr:.1f} ± {100*stderr:.1f}%")
print(f"written: {out}")
PY

echo "=== done $(date -Iseconds) ===" | tee -a "${LOG}"
