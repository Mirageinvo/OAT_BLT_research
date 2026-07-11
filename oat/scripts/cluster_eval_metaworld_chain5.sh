#!/usr/bin/env bash
# Paper-style MetaWorld MT4 eval (Appendix A): ONE best checkpoint,
# 5 sequential eval passes (eval seeds 0..4), each n_test=50 → 250 episodes total.
# Tasks are interleaved (mt4 suite); runner reports per-subtask SR in eval_log.json.
#
# Paper Table VI targets (OAT8, single sample):
#   box-close 44.4% | coffee-pull 26.4% | disassemble 17.2% | stick-pull 9.6% | avg 24.4%
#
# Usage (on cluster):
#   tmux new -s eval_mt4_chain5 -d 'bash scripts/cluster_eval_metaworld_chain5.sh'
# Optional:
#   CKPT=... OUT_ROOT=... GPU=0 bash scripts/cluster_eval_metaworld_chain5.sh
set -euo pipefail
cd /workspace/oat
source .venv/bin/activate

export OAT_USE_UV_RUN=0
GPU="${GPU:-0}"
# shellcheck disable=SC1091
source scripts/cluster_gpu_env.sh "${GPU}"

CKPT="${CKPT:-output/20260708/032431_train_oatpolicy_mw-mt4_N50/checkpoints/ep-0450_sr-0.280.ckpt}"
OUT_ROOT="${OUT_ROOT:-output/eval/metaworld_mt4_paper5_ep0450}"
LOG="${LOG:-logs/eval_mt4_chain5.log}"
N_TEST="${N_TEST:-50}"
BASE_SEED="${BASE_SEED:-1000}"
N_PARALLEL="${N_PARALLEL:-4}"
ENV_TASK="${ENV_TASK:-mt4}"

MT4_SUBTASKS=(box-close coffee-pull disassemble stick-pull)

mkdir -p logs "${OUT_ROOT}"

echo "=== MetaWorld MT4 chain eval | 5 seeds × n_test=${N_TEST} | OAT8 | task=${ENV_TASK} | $(date -Iseconds) ===" | tee "${LOG}"
echo "ckpt=${CKPT}" | tee -a "${LOG}"
echo "out_root=${OUT_ROOT}" | tee -a "${LOG}"
echo "gpu=${CUDA_VISIBLE_DEVICES} device=${OAT_DEVICE}" | tee -a "${LOG}"

if [[ ! -f "${CKPT}" ]]; then
  echo "ERROR: checkpoint not found: ${CKPT}" | tee -a "${LOG}"
  exit 1
fi

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
    --env_task_name "${ENV_TASK}" \
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
  echo ">>> seed ${EVAL_SEED} done: overall SR=${SR}" | tee -a "${LOG}"
done

python - <<PY | tee -a "${LOG}"
import json, math, pathlib

root = pathlib.Path("${OUT_ROOT}")
subtasks = ["box-close", "coffee-pull", "disassemble", "stick-pull"]

def task_key(task: str) -> str:
    return f"{task}/mean_success_rate_mean"

overall_srs = []
per_task = {t: [] for t in subtasks}

for seed in range(5):
    p = root / f"seed_{seed}" / "eval_log.json"
    with open(p) as f:
        d = json.load(f)
    overall_srs.append(d["mean_success_rate_mean"])
    for t in subtasks:
        k = task_key(t)
        if k in d:
            per_task[t].append(d[k])
        else:
            raise KeyError(f"missing {k} in {p}")

mean_overall = sum(overall_srs) / len(overall_srs)
stderr_overall = (
    math.sqrt(sum((x - mean_overall) ** 2 for x in overall_srs) / (len(overall_srs) - 1))
    / math.sqrt(len(overall_srs))
    if len(overall_srs) > 1
    else 0.0
)

task_means = {}
task_stderrs = {}
for t in subtasks:
    vals = per_task[t]
    m = sum(vals) / len(vals)
    se = (
        math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) / math.sqrt(len(vals))
        if len(vals) > 1
        else 0.0
    )
    task_means[t] = m
    task_stderrs[t] = se

avg_of_tasks = sum(task_means.values()) / len(task_means)

paper = {
    "box-close": 0.444,
    "coffee-pull": 0.264,
    "disassemble": 0.172,
    "stick-pull": 0.096,
    "average": 0.244,
}

summary = {
    "checkpoint": "${CKPT}",
    "env_task_name": "${ENV_TASK}",
    "n_test_per_seed": ${N_TEST},
    "base_seed": ${BASE_SEED},
    "per_seed_overall_sr": {str(i): overall_srs[i] for i in range(5)},
    "per_task_per_seed_sr": {t: {str(i): per_task[t][i] for i in range(5)} for t in subtasks},
    "mean_success_rate_overall": mean_overall,
    "stderr_overall": stderr_overall,
    "per_task_mean_sr": task_means,
    "per_task_stderr": task_stderrs,
    "mean_of_task_means": avg_of_tasks,
    "paper_table_vi_targets": paper,
    "pct": {
        "overall": {"mean": 100 * mean_overall, "stderr": 100 * stderr_overall},
        "per_task": {t: 100 * task_means[t] for t in subtasks},
        "mean_of_tasks": 100 * avg_of_tasks,
    },
}
out = root / "summary.json"
out.write_text(json.dumps(summary, indent=2))

print("")
print("=== FINAL MetaWorld MT4 (5 eval seeds, 250 episodes, interleaved) ===")
print(f"per-seed overall SR: {overall_srs}")
print(f"overall mean ± stderr: {100*mean_overall:.1f} ± {100*stderr_overall:.1f}%")
print("")
print("Per-task mean SR (paper Table VI in parentheses):")
for t in subtasks:
    print(f"  {t:14s}: {100*task_means[t]:5.1f}% ± {100*task_stderrs[t]:4.1f}%   (paper {100*paper[t]:.1f}%)")
print(f"  {'average (task mean)':14s}: {100*avg_of_tasks:5.1f}%              (paper {100*paper['average']:.1f}%)")
print(f"written: {out}")
PY

echo "=== MT4 chain eval complete | $(date -Iseconds) ===" | tee -a "${LOG}"
