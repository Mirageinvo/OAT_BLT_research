#!/usr/bin/env bash
# Wait for the in-flight coffee-pull train-time eval to finish, lock Top-1,
# stop the train (same as other MW after TopK lock), then Wave1 → Wave2 matched.
#
# Protocol (aligned with stick/box/disassemble):
#   TEST_START_SEED=10000  n_test=50  n_exp=5  OAT8  BoN N=8 vote
#   Wave1 SKIP_AWR=1 → matched_s10000/coffee-pull/
#   Wave2 AWR → awr_s10000_coffee-pull.{npz,ckpt}
set -euo pipefail
cd /workspace/oat

export MUJOCO_GL=egl
export OAT_USE_UV_RUN=0
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

GPU="${GPU:-1}"
RUN_DIR="${RUN_DIR:-output/20260720/090816_train_oatpolicy_mw-coffee-pull_st_N50}"
CKPT_DIR="${RUN_DIR}/checkpoints"
LOG="${LOG:-logs/coffee_pull_matched_after_eval_gpu${GPU}.log}"
mkdir -p logs

best_topk() {
  python3 - <<'PY'
import re
from pathlib import Path
import os
ckpt_dir = Path(os.environ["CKPT_DIR"])
best = None  # (sr, ep, path)
for p in ckpt_dir.glob("ep-*_sr-*.ckpt"):
    m = re.match(r"ep-(\d+)_sr-([0-9.]+)\.ckpt$", p.name)
    if not m:
        continue
    ep, sr = int(m.group(1)), float(m.group(2))
    if best is None or sr > best[0] or (sr == best[0] and ep > best[1]):
        best = (sr, ep, str(p))
if best is None:
    raise SystemExit("no TopK ckpts in " + str(ckpt_dir))
print(f"{best[2]}\t{best[0]}\t{best[1]}")
PY
}

eval_active() {
  # True while the train pane is still inside a MetaWorld coffee-pull Eval progress bar.
  tmux has-session -t coffee_policy_refit 2>/dev/null || return 1
  local pane
  pane="$(tmux capture-pane -t coffee_policy_refit -p 2>/dev/null | tr '\r' '\n' | tail -30)"
  echo "${pane}" | grep -qE 'Eval oatpolicy_rgb in MetaWorld::coffee-pull [0-9]+/63'
}

snapshot_topk_names() {
  ls -1 "${CKPT_DIR}"/ep-*_sr-*.ckpt 2>/dev/null | sort || true
}

export CKPT_DIR
{
  echo "=== coffee-pull wait-eval → TopK lock → matched $(date -Iseconds) ==="
  echo "run_dir=${RUN_DIR}"
  echo "gpu=${GPU}"
  BEFORE="$(best_topk)"
  BEFORE_PATH="$(echo "${BEFORE}" | cut -f1)"
  BEFORE_SR="$(echo "${BEFORE}" | cut -f2)"
  BEFORE_EP="$(echo "${BEFORE}" | cut -f3)"
  echo "topk_before path=${BEFORE_PATH} sr=${BEFORE_SR} ep=${BEFORE_EP}"
  BEFORE_NAMES="$(snapshot_topk_names)"
  echo "topk_names_before:"
  echo "${BEFORE_NAMES}"
} | tee "${LOG}"

# --- wait until current train-time eval leaves the pane ---
echo "[wait] for train-time eval to finish…" | tee -a "${LOG}"
if ! tmux has-session -t coffee_policy_refit 2>/dev/null; then
  echo "[warn] coffee_policy_refit already gone — proceed with current TopK" | tee -a "${LOG}"
else
  # Require eval inactive for 3 consecutive polls (~90s) so inter-chunk gaps don't false-trigger.
  inactive_streak=0
  saw_eval=0
  for i in $(seq 1 240); do  # up to ~2h @ 30s
    if eval_active; then
      saw_eval=1
      inactive_streak=0
    else
      if (( saw_eval == 1 )); then
        inactive_streak=$((inactive_streak + 1))
      else
        # Haven't observed Eval yet — keep waiting (eval may still be starting / pane lag).
        inactive_streak=0
      fi
    fi
    if (( saw_eval == 1 && inactive_streak >= 3 )); then
      echo "[wait] eval finished (poll=${i}, inactive_streak=${inactive_streak}) $(date -Iseconds)" | tee -a "${LOG}"
      break
    fi
    if (( i % 4 == 0 )); then
      pane_tail="$(tmux capture-pane -t coffee_policy_refit -p 2>/dev/null | tr '\r' '\n' | grep -E 'Eval oatpolicy_rgb|Training epoch|mean_success' | tail -3 || true)"
      echo "[poll ${i}] saw_eval=${saw_eval} inactive=${inactive_streak} $(date -Iseconds) ${pane_tail}" | tee -a "${LOG}"
    fi
    sleep 30
  done
  # Settle: TopK write can lag after the eval bar vanishes.
  sleep 45
  # If a brand-new TopK file is still growing, wait until stable.
  for j in 1 2 3 4 5 6; do
    s1="$(snapshot_topk_names)"
    sleep 10
    s2="$(snapshot_topk_names)"
    [[ "${s1}" == "${s2}" ]] && break
    echo "[wait] TopK set still changing… (${j})" | tee -a "${LOG}"
  done
fi

AFTER="$(best_topk)"
AFTER_PATH="$(echo "${AFTER}" | cut -f1)"
AFTER_SR="$(echo "${AFTER}" | cut -f2)"
AFTER_EP="$(echo "${AFTER}" | cut -f3)"
AFTER_NAMES="$(snapshot_topk_names)"
{
  echo "topk_after path=${AFTER_PATH} sr=${AFTER_SR} ep=${AFTER_EP}"
  echo "topk_names_after:"
  echo "${AFTER_NAMES}"
} | tee -a "${LOG}"

# Decision: if a new/better TopK appeared use it; else keep previous Top-1.
BASE_CKPT="${AFTER_PATH}"
if python3 - <<PY
before=${BEFORE_SR}
after=${AFTER_SR}
raise SystemExit(0 if after > before + 1e-9 else 1)
PY
then
  echo "DECISION: NEW TopK is best → use ep-${AFTER_EP} sr=${AFTER_SR}" | tee -a "${LOG}"
else
  echo "DECISION: eval did not beat Top-1 → use ep-${AFTER_EP} sr=${AFTER_SR} (current TopK lock)" | tee -a "${LOG}"
fi
echo "BASE_CKPT=${BASE_CKPT}" | tee -a "${LOG}"
[[ -f "${BASE_CKPT}" ]] || { echo "ERROR missing ${BASE_CKPT}"; exit 1; }

# Point paper defaults at the locked ckpt via symlink-friendly absolute path string.
# Stop train so GPU1 is free for matched (same as other MW after TopK lock).
if tmux has-session -t coffee_policy_refit 2>/dev/null; then
  echo "[stop] killing coffee_policy_refit after TopK lock $(date -Iseconds)" | tee -a "${LOG}"
  tmux kill-session -t coffee_policy_refit || true
  sleep 5
fi

# Persist lock pointer for RESULTS / later rematches.
LOCK_FILE="my_models/coffee_pull_topk_lock.txt"
mkdir -p my_models
{
  echo "base_ckpt=${BASE_CKPT}"
  echo "sr=${AFTER_SR}"
  echo "ep=${AFTER_EP}"
  echo "run_dir=${RUN_DIR}"
  echo "locked_at=$(date -Iseconds)"
} | tee "${LOCK_FILE}" | tee -a "${LOG}"

# --- Wave1: baseline + BoN ---
WAVE1_LOG="logs/matched_s10000_coffee-pull_gpu${GPU}.log"
echo "=== Wave1 START $(date -Iseconds) ===" | tee -a "${LOG}"
SUITE=coffee-pull GPU="${GPU}" \
  BASE_CKPT="${BASE_CKPT}" \
  LOG="${WAVE1_LOG}" \
  SKIP_AWR=1 FORCE_RERUN=1 \
  TEST_START_SEED=10000 N_EXP=5 N_PARALLEL=4 \
  bash scripts/cluster_matched_triplet.sh
echo "=== Wave1 DONE $(date -Iseconds) ===" | tee -a "${LOG}"

# --- Wave2: AWR collect/train/eval ---
WAVE2_LOG="logs/awr_s10000_coffee-pull_wave2_gpu${GPU}.log"
echo "=== Wave2 START $(date -Iseconds) ===" | tee -a "${LOG}"
SUITE=coffee-pull GPU="${GPU}" \
  BASE_CKPT="${BASE_CKPT}" \
  WAVE1_ROOT=output/eval/matched_s10000/coffee-pull \
  FORCE_RECOLLECT=1 \
  LOG="${WAVE2_LOG}" \
  TEST_START_SEED=10000 N_EXP=5 \
  bash scripts/cluster_matched_paper_wave2_awr.sh
echo "=== coffee-pull Wave1+Wave2 ALL DONE $(date -Iseconds) ===" | tee -a "${LOG}"
