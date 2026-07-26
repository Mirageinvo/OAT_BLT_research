#!/usr/bin/env bash
# Paper matched triplet: baseline / BoN N=8 / (optional) AWR
# on a SHARED fixed init set that is NOT the train selection pool.
#
# Paper defaults (override with env):
#   TEST_START_SEED=10000  n_test=50  num_exp=5  T=1 topk=10  OAT8  BoN N=8 vote
#   OUT_ROOT=output/eval/matched_s${TEST_START_SEED}/<suite>/
#
# Selection pool during training stays at seed 1000 — do NOT report that as held-out.
#
# Usage:
#   SUITE=can GPU=0 bash scripts/cluster_matched_triplet.sh
#   SKIP_AWR=1 SUITE=box-close GPU=1 bash scripts/cluster_matched_triplet.sh
#   AWR_CKPT=my_models/awr_s10000_can.ckpt SUITE=can GPU=0 bash scripts/cluster_matched_triplet.sh
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
source .venv/bin/activate
export MUJOCO_GL=egl
export OAT_USE_UV_RUN=0

SUITE="${SUITE:?Set SUITE=lift|can|square|mt4|coffee-pull|stick-pull|disassemble|box-close}"
GPU="${GPU:-0}"
# shellcheck disable=SC1091
source scripts/cluster_gpu_env.sh "${GPU}"
bash scripts/patch_robosuite_egl_assert.sh

N_TEST="${N_TEST:-50}"
TEST_START_SEED="${TEST_START_SEED:-10000}"   # paper report pool (disjoint from selection 1000)
N_EXP="${N_EXP:-5}"
N_PARALLEL="${N_PARALLEL:-4}"
BON_N="${BON_N:-8}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOPK="${TOPK:-10}"
SKIP_AWR="${SKIP_AWR:-0}"                     # 1 = baseline+BoN only
SKIP_BASELINE_BON="${SKIP_BASELINE_BON:-0}"   # 1 = Wave2: keep existing baseline/BoN, only AWR(+summary)
FORCE_RERUN="${FORCE_RERUN:-1}"              # 1 = never skip baseline (paper-safe)

case "${SUITE}" in
  lift)
    # Default still ep-0900 for legacy; paper TopK lock = ep-1400 (override BASE_CKPT / use _launch_lift_ep1400_matched.sh).
    BASE_CKPT="${BASE_CKPT:-output/20260719/144024_train_oatpolicy_lift_N200/checkpoints/ep-0900_sr-0.930.ckpt}"
    AWR_CKPT="${AWR_CKPT:-}"
    ENV_TASK=""
    SUITE_DIR="lift"
    ;;
  can)
    BASE_CKPT="${BASE_CKPT:-output/20260706/173343_train_oatpolicy_can_N200/checkpoints/ep-1700_sr-0.940.ckpt}"
    AWR_CKPT="${AWR_CKPT:-}"
    ENV_TASK=""
    SUITE_DIR="can"
    ;;
  square)
    # Prefer live TopK lock (scripts/_launch_square_matched_on_plateau.sh); else override BASE_CKPT.
    BASE_CKPT="${BASE_CKPT:-output/20260720/215024_train_oatpolicy_square_N200/checkpoints/ep-0700_sr-0.420.ckpt}"
    AWR_CKPT="${AWR_CKPT:-}"
    ENV_TASK=""
    SUITE_DIR="square"
    ;;
  mt4)
    echo "ERROR: SUITE=mt4 removed from paper track (policy run deleted 2026-07-22). Use MW specialists." >&2
    exit 1
    ;;
  coffee-pull)
    # Paper refit 20260720; override via BASE_CKPT= (TopK lock after train-eval).
    BASE_CKPT="${BASE_CKPT:-output/20260720/090816_train_oatpolicy_mw-coffee-pull_st_N50/checkpoints/ep-1000_sr-0.432.ckpt}"
    AWR_CKPT="${AWR_CKPT:-}"
    ENV_TASK="coffee-pull"
    SUITE_DIR="coffee-pull"
    ;;
  stick-pull)
    BASE_CKPT="${BASE_CKPT:-output/20260711/134439_train_oatpolicy_mw-stick-pull_st_N50/checkpoints/ep-0800_sr-0.212.ckpt}"
    AWR_CKPT="${AWR_CKPT:-}"
    ENV_TASK="stick-pull"
    SUITE_DIR="stick-pull"
    ;;
  disassemble)
    BASE_CKPT="${BASE_CKPT:-output/20260711/134440_train_oatpolicy_mw-disassemble_st_N50/checkpoints/ep-1400_sr-0.700.ckpt}"
    AWR_CKPT="${AWR_CKPT:-}"
    ENV_TASK="disassemble"
    SUITE_DIR="disassemble"
    ;;
  box-close)
    BASE_CKPT="${BASE_CKPT:-output/20260711/134439_train_oatpolicy_mw-box-close_st_N50/checkpoints/ep-2000_sr-0.552.ckpt}"
    AWR_CKPT="${AWR_CKPT:-}"
    ENV_TASK="box-close"
    # keep one path under the paper root (no metaworld_ prefix confusion)
    SUITE_DIR="box-close"
    ;;
  *)
    echo "Unknown SUITE=${SUITE}" >&2
    exit 1
    ;;
esac

OUT_ROOT="${OUT_ROOT:-output/eval/matched_s${TEST_START_SEED}/${SUITE_DIR}}"
LOG="${LOG:-logs/matched_s${TEST_START_SEED}_${SUITE_DIR}_gpu${GPU}.log}"
mkdir -p logs "${OUT_ROOT}"

env_args=()
if [[ -n "${ENV_TASK}" ]]; then
  env_args+=(--env_task_name "${ENV_TASK}")
fi

run_eval() {
  local mode="$1" ckpt="$2" out="$3"
  shift 3 || true
  local extra=("$@")
  rm -rf "${out}"
  echo "" | tee -a "${LOG}"
  echo "[$(date -Iseconds)] START ${mode} -> ${out}" | tee -a "${LOG}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl \
    python scripts/eval_policy_sim.py \
      -c "${ckpt}" \
      -o "${out}" \
      -d "${OAT_DEVICE}" \
      -n "${N_EXP}" \
      --n_test "${N_TEST}" \
      --test_start_seed "${TEST_START_SEED}" \
      --n_parallel_envs "${N_PARALLEL}" \
      --use_k_tokens 8 \
      --entropy_threshold 0 \
      --temperature "${TEMPERATURE}" \
      --topk "${TOPK}" \
      "${env_args[@]}" \
      "${extra[@]}" \
    2>&1 | tee -a "${LOG}"
  echo "[$(date -Iseconds)] DONE ${mode}" | tee -a "${LOG}"
}

echo "=== PAPER MATCHED TRIPLET ${SUITE} ===" | tee "${LOG}"
echo "gpu=${CUDA_VISIBLE_DEVICES} device=${OAT_DEVICE}" | tee -a "${LOG}"
echo "base=${BASE_CKPT}" | tee -a "${LOG}"
echo "awr=${AWR_CKPT:-NONE} skip_awr=${SKIP_AWR}" | tee -a "${LOG}"
echo "n_test=${N_TEST} paper_seed=${TEST_START_SEED} n_exp=${N_EXP} T=${TEMPERATURE} topk=${TOPK}" | tee -a "${LOG}"
echo "out_root=${OUT_ROOT}" | tee -a "${LOG}"
echo "NOTE: train selection pool is seed 1000; this report pool must stay disjoint." | tee -a "${LOG}"

[[ -f "${BASE_CKPT}" ]] || { echo "ERROR missing ${BASE_CKPT}" | tee -a "${LOG}"; exit 1; }

BASE_OUT="${OUT_ROOT}/baseline_n${N_EXP}"
BON_OUT="${OUT_ROOT}/bon_n${BON_N}_n${N_EXP}"
AWR_OUT="${OUT_ROOT}/awr_n${N_EXP}"

# Wave2: never wipe finished Wave1 baseline/BoN
if [[ "${SKIP_BASELINE_BON}" == "1" ]]; then
  [[ -f "${BASE_OUT}/eval_log.json" ]] || { echo "ERROR SKIP_BASELINE_BON=1 but missing ${BASE_OUT}/eval_log.json" | tee -a "${LOG}"; exit 1; }
  [[ -f "${BON_OUT}/eval_log.json" ]] || { echo "ERROR SKIP_BASELINE_BON=1 but missing ${BON_OUT}/eval_log.json" | tee -a "${LOG}"; exit 1; }
  echo "[skip] baseline+BoN (SKIP_BASELINE_BON=1 Wave2; keep paper Wave1)" | tee -a "${LOG}"
else
  # Paper-safe: only skip baseline if FORCE_RERUN=0 AND existing log matches this seed
  skip_baseline=0
  if [[ "${FORCE_RERUN}" != "1" && -f "${BASE_OUT}/eval_log.json" && -f "${OUT_ROOT}/summary.json" ]]; then
    prev_seed="$(python -c "import json; print(json.load(open('${OUT_ROOT}/summary.json')).get('protocol',{}).get('test_start_seed',''))" 2>/dev/null || true)"
    if [[ "${prev_seed}" == "${TEST_START_SEED}" ]]; then
      skip_baseline=1
    fi
  fi

  if [[ "${skip_baseline}" == "1" ]]; then
    echo "[skip] baseline already matched protocol seed=${TEST_START_SEED}" | tee -a "${LOG}"
  else
    run_eval baseline "${BASE_CKPT}" "${BASE_OUT}"
  fi

  run_eval bon "${BASE_CKPT}" "${BON_OUT}" \
    --bon_free "${BON_N}" --bon_signal vote
fi

if [[ "${SKIP_AWR}" == "1" ]]; then
  echo "[skip] AWR eval — SKIP_AWR=1 (baseline+BoN paper wave)" | tee -a "${LOG}"
elif [[ -n "${AWR_CKPT}" && -f "${AWR_CKPT}" ]]; then
  run_eval awr "${AWR_CKPT}" "${AWR_OUT}"
else
  echo "[skip] AWR eval — no AWR_CKPT (train a paper AWR first, then re-run with AWR_CKPT=...)" | tee -a "${LOG}"
fi

python - <<PY | tee -a "${LOG}"
import json, pathlib
root = pathlib.Path("${OUT_ROOT}")
suite = "${SUITE}"
n_exp = int("${N_EXP}")
bon_n = int("${BON_N}")

def load(p):
    d = json.load(open(p))
    # prefer flat key; fall back to nested "<task>/mean_success_rate_mean"
    if "mean_success_rate_mean" in d:
        m = float(d["mean_success_rate_mean"])
        s = float(d.get("mean_success_rate_std", 0.0))
    else:
        mk = [k for k in d if str(k).endswith("mean_success_rate_mean")]
        if not mk:
            raise KeyError(f"no success key in {p}")
        m = float(d[mk[0]])
        s = float(d.get(mk[0].replace("_mean", "_std"), 0.0))
    return m, s

summary = {
    "suite": suite,
    "suite_dir": "${SUITE_DIR}",
    "protocol": {
        "test_start_seed": int("${TEST_START_SEED}"),
        "selection_seed_note": "train TopK used seed 1000; paper report must use this disjoint pool",
        "n_test": int("${N_TEST}"),
        "num_exp": n_exp,
        "use_k_tokens": 8,
        "entropy_threshold": 0,
        "temperature": float("${TEMPERATURE}"),
        "topk": int("${TOPK}"),
        "bon_free": bon_n,
        "bon_signal": "vote",
        "episodes": f"{int('${TEST_START_SEED}')}..{int('${TEST_START_SEED}')+int('${N_TEST}')-1}",
    },
    "base_ckpt": "${BASE_CKPT}",
    "awr_ckpt": "${AWR_CKPT}" or None,
    "artifacts": {
        "root": str(root),
        "baseline": str(root / f"baseline_n{n_exp}"),
        "bon": str(root / f"bon_n{bon_n}_n{n_exp}"),
        "awr": str(root / f"awr_n{n_exp}"),
        "log": "${LOG}",
        "summary": str(root / "summary.json"),
    },
}
for name, path in (
    (f"baseline_n{n_exp}", root / f"baseline_n{n_exp}" / "eval_log.json"),
    (f"bon_n{bon_n}_n{n_exp}", root / f"bon_n{bon_n}_n{n_exp}" / "eval_log.json"),
    (f"awr_n{n_exp}", root / f"awr_n{n_exp}" / "eval_log.json"),
):
    if path.exists():
        m, s = load(path)
        summary[name] = {"mean": m, "std": s, "pct": f"{100*m:.1f} ± {100*s:.1f}%"}
    else:
        summary[name] = None

b = summary.get(f"baseline_n{n_exp}")
if b:
    for method in (f"bon_n{bon_n}_n{n_exp}", f"awr_n{n_exp}"):
        m = summary.get(method)
        if m:
            summary[f"delta_{method}_pp"] = 100.0 * (m["mean"] - b["mean"])

out = root / "summary.json"
out.write_text(json.dumps(summary, indent=2))
print("=== PAPER MATCHED SUMMARY ===")
print(json.dumps(summary, indent=2))
print("written", out)
PY

echo "=== ALL DONE ${SUITE} $(date -Iseconds) ===" | tee -a "${LOG}"
