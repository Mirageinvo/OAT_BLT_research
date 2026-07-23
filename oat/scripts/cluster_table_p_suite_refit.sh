#!/usr/bin/env bash
# Full Table P refit for one suite: policy train → Wave1 (baseline+BoN) → Wave2 (AWR) → latency (C + C′).
#
# Keeps frozen tokenizer ckpt; retrains policy + all matched_s10000 paper artifacts.
#
# Usage (inside docker /workspace/oat):
#   SUITE=square GPU=0 bash scripts/cluster_table_p_suite_refit.sh
#   SUITE=coffee-pull GPU=1 bash scripts/cluster_table_p_suite_refit.sh
#
# Env:
#   WAIT_GPU=1          poll until chosen GPU has <3GB used (default 1)
#   TEST_START_SEED=10000
set -euo pipefail
cd /workspace/oat
source .venv/bin/activate
export MUJOCO_GL=egl
export OAT_USE_UV_RUN=0

SUITE="${SUITE:?Set SUITE=square|coffee-pull}"
GPU="${GPU:-0}"
WAIT_GPU="${WAIT_GPU:-1}"
TEST_START_SEED="${TEST_START_SEED:-10000}"

wait_for_gpu() {
  local g="$1"
  [[ "${WAIT_GPU}" == "1" ]] || return 0
  echo "=== waiting for GPU${g} (<3000 MiB used) ==="
  while true; do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${g}" 2>/dev/null | tr -d ' ')"
    if [[ -n "${used}" && "${used}" -lt 3000 ]]; then
      echo "GPU${g} free enough (${used} MiB)"
      return 0
    fi
    echo "GPU${g} busy (${used:-?} MiB) — sleep 120s"
    sleep 120
  done
}

latest_policy_run() {
  local pattern="$1"
  python - <<PY
import pathlib
pat = "${pattern}"
root = pathlib.Path("output")
cands = [p for p in root.glob("**/*") if p.is_dir() and pat in p.name]
if not cands:
    raise SystemExit(f"No run dir matching {pat!r}")
cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
print(cands[0])
PY
}

pick_best_sr_ckpt() {
  local run_dir="$1"
  python scripts/select_best_ckpt_by_name.py --run-dir "${run_dir}" --metric sr --mode max --top 1
}

run_wave1() {
  local suite="$1" base_ckpt="$2"
  SUITE="${suite}" GPU="${GPU}" BASE_CKPT="${base_ckpt}" \
    TEST_START_SEED="${TEST_START_SEED}" SKIP_AWR=1 FORCE_RERUN=1 \
    LOG="logs/matched_s${TEST_START_SEED}_${suite}_refit_gpu${GPU}.log" \
    bash scripts/cluster_matched_triplet.sh
}

run_wave2() {
  local suite="$1" base_ckpt="$2"
  SUITE="${suite}" GPU="${GPU}" BASE_CKPT="${base_ckpt}" \
    TEST_START_SEED="${TEST_START_SEED}" FORCE_RECOLLECT=1 \
    bash scripts/cluster_matched_paper_wave2_awr.sh
}

run_latency() {
  local suite="$1"
  if [[ -z "${OAT_GIT_COMMIT:-}" ]]; then
    echo "WARN: OAT_GIT_COMMIT unset — skipping latency (set before launch for Table C)"
    return 0
  fi
  SUITES="${suite}" GPU="${GPU}" FAIR_KV=0 \
    bash scripts/cluster_latency_paper_done.sh
  SUITES="${suite}" GPU="${GPU}" FAIR_KV=1 \
    bash scripts/cluster_latency_paper_done.sh
}

LOG="logs/table_p_refit_${SUITE}_gpu${GPU}.log"
mkdir -p logs
exec > >(tee -a "${LOG}") 2>&1

echo "=== TABLE P REFIT ${SUITE} $(date -Iseconds) gpu=${GPU} ==="
wait_for_gpu "${GPU}"

case "${SUITE}" in
  square)
    SQUARE_TOK="${SQUARE_TOK:-output/20260706/005048_train_oattok_square_N200/checkpoints/ep-0690_mse-0.004.ckpt}"
    [[ -f "${SQUARE_TOK}" ]] || { echo "ERROR missing tokenizer ${SQUARE_TOK}"; exit 1; }
    echo "[POL] fresh square policy train"
    RESUME_RUN_DIR= SQUARE_TOK="${SQUARE_TOK}" GPU="${GPU}" \
      LOG="logs/refit_train_policy_square_s42_gpu${GPU}.log" \
      bash scripts/cluster_policy_square_paper.sh
    POL_RUN="$(latest_policy_run train_oatpolicy_square_N200)"
    BASE_CKPT="$(pick_best_sr_ckpt "${POL_RUN}")"
    ;;
  coffee-pull)
    COFFEE_TOK="${COFFEE_TOK:-output/20260710/212943_train_oattok_mw-coffee-pull_st_N50/checkpoints/ep-2670_mse-0.039.ckpt}"
    [[ -f "${COFFEE_TOK}" ]] || { echo "ERROR missing tokenizer ${COFFEE_TOK}"; exit 1; }
    echo "[POL] fresh coffee-pull policy train"
    TASK=coffee-pull GPU="${GPU}" SEED=0 NUM_DEMO=50 TOKENIZER_CKPT="${COFFEE_TOK}" \
      bash scripts/cluster_policy_metaworld_single.sh
    POL_RUN="$(latest_policy_run train_oatpolicy_mw-coffee-pull_st_N50)"
    BASE_CKPT="$(pick_best_sr_ckpt "${POL_RUN}")"
    ;;
  *)
    echo "Unknown SUITE=${SUITE}" >&2
    exit 1
    ;;
esac

echo "[POL] run_dir=${POL_RUN}"
echo "[POL] base_ckpt=${BASE_CKPT}"
[[ -f "${BASE_CKPT}" ]] || { echo "ERROR missing ${BASE_CKPT}"; exit 1; }

echo "[WAVE1] baseline + BoN @ seed ${TEST_START_SEED}"
run_wave1 "${SUITE}" "${BASE_CKPT}"

echo "[WAVE2] AWR collect/train/eval"
run_wave2 "${SUITE}" "${BASE_CKPT}"

echo "[LATENCY] Table C + C′"
run_latency "${SUITE}"

echo "=== TABLE P REFIT DONE ${SUITE} $(date -Iseconds) ==="
echo "policy_run=${POL_RUN}"
echo "base_ckpt=${BASE_CKPT}"
echo "artifacts=output/eval/matched_s${TEST_START_SEED}/${SUITE}/"
