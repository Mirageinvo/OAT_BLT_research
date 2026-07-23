#!/usr/bin/env bash
# Square paper matched: wait for train TopK plateau → lock Top-1 → Wave1 → Wave2.
#
# Plateau rule (same as box-close): after the latest peak SR, the next
# N_BELOW (=2) train-eval checkpoints are strictly below that peak → stop.
#
# Protocol (aligned with can / stick / box):
#   TEST_START_SEED=10000  n_test=50  n_exp=5  OAT8  BoN N=8 vote
#   Wave1 SKIP_AWR=1 → matched_s10000/square/
#   Wave2 AWR → awr_s10000_square.{npz,ckpt}
#
# Usage (cluster docker):
#   GPU=0 bash scripts/_launch_square_matched_on_plateau.sh
set -euo pipefail
cd /workspace/oat

export MUJOCO_GL=egl
export OAT_USE_UV_RUN=0
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

GPU="${GPU:-0}"
RUN_DIR="${RUN_DIR:-output/20260720/215024_train_oatpolicy_square_N200}"
CKPT_DIR="${RUN_DIR}/checkpoints"
TRAIN_TMUX="${TRAIN_TMUX:-square_policy_resume}"
N_BELOW="${N_BELOW:-2}"          # consecutive evals below peak → plateau
POLL_SEC="${POLL_SEC:-60}"
LOG="${LOG:-logs/square_matched_on_plateau_gpu${GPU}.log}"
LOCK_FILE="${LOCK_FILE:-my_models/square_topk_lock.txt}"
mkdir -p logs my_models

export CKPT_DIR
export N_BELOW

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

plateau_status() {
  # Prints: PLATEAU|WAITING <best_sr> <best_ep> <n_below> <last_ep> <last_sr>
  # Rule: after the latest peak epoch, the next N_BELOW train-evals are all < peak.
  python3 - <<'PY'
import re, os
from pathlib import Path
ckpt_dir = Path(os.environ["CKPT_DIR"])
n_below_need = int(os.environ["N_BELOW"])
rows = []
for p in ckpt_dir.glob("ep-*_sr-*.ckpt"):
    m = re.match(r"ep-(\d+)_sr-([0-9.]+)\.ckpt$", p.name)
    if not m:
        continue
    rows.append((int(m.group(1)), float(m.group(2))))
rows.sort()
if not rows:
    print("WAITING\t0\t0\t0\t0\t0")
    raise SystemExit(0)
best_sr = max(sr for _, sr in rows)
best_ep = max(ep for ep, sr in rows if abs(sr - best_sr) < 1e-9)
last_ep, last_sr = rows[-1]
post = [(ep, sr) for ep, sr in rows if ep > best_ep]
if len(post) < n_below_need:
    status, below = "WAITING", len(post)
elif all(sr < best_sr - 1e-9 for _, sr in post[:n_below_need]):
    status, below = "PLATEAU", n_below_need
else:
    below = 0
    for _, sr in post:
        if sr < best_sr - 1e-9:
            below += 1
        else:
            break
    status = "WAITING"
print(f"{status}\t{best_sr}\t{best_ep}\t{below}\t{last_ep}\t{last_sr}")
PY
}

eval_active() {
  tmux has-session -t "${TRAIN_TMUX}" 2>/dev/null || return 1
  local pane
  pane="$(tmux capture-pane -t "${TRAIN_TMUX}" -p 2>/dev/null | tr '\r' '\n' | tail -40)"
  echo "${pane}" | grep -qE 'Eval oatpolicy_rgb in RoboMimic::square [0-9]+/'
}

ckpt_fingerprint() {
  ls -1 "${CKPT_DIR}"/ep-*_sr-*.ckpt 2>/dev/null | sort | md5sum | awk '{print $1}'
}

{
  echo "=== square plateau→matched START $(date -Iseconds) ==="
  echo "run_dir=${RUN_DIR}"
  echo "train_tmux=${TRAIN_TMUX}"
  echo "n_below=${N_BELOW} poll=${POLL_SEC}s gpu=${GPU}"
  echo "topk_now=$(best_topk)"
} | tee "${LOG}"

prev_fp="$(ckpt_fingerprint || true)"
while true; do
  if ! tmux has-session -t "${TRAIN_TMUX}" 2>/dev/null; then
    echo "[warn] ${TRAIN_TMUX} gone — lock current Top-1 and proceed $(date -Iseconds)" | tee -a "${LOG}"
    break
  fi
  if eval_active; then
    pane_tail="$(tmux capture-pane -t "${TRAIN_TMUX}" -p 2>/dev/null | tr '\r' '\n' | grep -E 'Eval oatpolicy_rgb|Training epoch' | tail -2 || true)"
    echo "[wait] train-eval active $(date -Iseconds) ${pane_tail}" | tee -a "${LOG}"
    sleep "${POLL_SEC}"
    continue
  fi
  # settle after eval bar disappears so new TopK file can land
  sleep 20
  fp="$(ckpt_fingerprint || true)"
  if [[ "${fp}" != "${prev_fp}" ]]; then
    echo "[ckpt] set changed ${prev_fp} → ${fp} $(date -Iseconds)" | tee -a "${LOG}"
    prev_fp="${fp}"
    # wait until stable
    for _ in 1 2 3 4 5; do
      sleep 10
      fp2="$(ckpt_fingerprint || true)"
      [[ "${fp2}" == "${fp}" ]] && break
      fp="${fp2}"
    done
    prev_fp="${fp}"
  fi

  status_line="$(plateau_status)"
  echo "[status] ${status_line} $(date -Iseconds)" | tee -a "${LOG}"
  status="$(echo "${status_line}" | cut -f1)"
  if [[ "${status}" == "PLATEAU" ]]; then
    echo "[plateau] detected — locking Top-1 $(date -Iseconds)" | tee -a "${LOG}"
    break
  fi
  sleep "${POLL_SEC}"
done

AFTER="$(best_topk)"
BASE_CKPT="$(echo "${AFTER}" | cut -f1)"
AFTER_SR="$(echo "${AFTER}" | cut -f2)"
AFTER_EP="$(echo "${AFTER}" | cut -f3)"
{
  echo "BASE_CKPT=${BASE_CKPT}"
  echo "sr=${AFTER_SR} ep=${AFTER_EP}"
} | tee -a "${LOG}"
[[ -f "${BASE_CKPT}" ]] || { echo "ERROR missing ${BASE_CKPT}"; exit 1; }

{
  echo "base_ckpt=${BASE_CKPT}"
  echo "sr=${AFTER_SR}"
  echo "ep=${AFTER_EP}"
  echo "run_dir=${RUN_DIR}"
  echo "locked_at=$(date -Iseconds)"
  echo "rule=peak_then_${N_BELOW}_below"
} | tee "${LOCK_FILE}" | tee -a "${LOG}"

if tmux has-session -t "${TRAIN_TMUX}" 2>/dev/null; then
  echo "[stop] killing ${TRAIN_TMUX} after TopK lock $(date -Iseconds)" | tee -a "${LOG}"
  tmux kill-session -t "${TRAIN_TMUX}" || true
  sleep 5
fi

WAVE1_LOG="logs/matched_s10000_square_gpu${GPU}.log"
echo "=== Wave1 START $(date -Iseconds) ===" | tee -a "${LOG}"
SUITE=square GPU="${GPU}" \
  BASE_CKPT="${BASE_CKPT}" \
  LOG="${WAVE1_LOG}" \
  SKIP_AWR=1 FORCE_RERUN=1 \
  TEST_START_SEED=10000 N_EXP=5 N_PARALLEL=4 \
  bash scripts/cluster_matched_triplet.sh
echo "=== Wave1 DONE $(date -Iseconds) ===" | tee -a "${LOG}"

WAVE2_LOG="logs/awr_s10000_square_wave2_gpu${GPU}.log"
echo "=== Wave2 START $(date -Iseconds) ===" | tee -a "${LOG}"
SUITE=square GPU="${GPU}" \
  BASE_CKPT="${BASE_CKPT}" \
  WAVE1_ROOT=output/eval/matched_s10000/square \
  FORCE_RECOLLECT=1 \
  LOG="${WAVE2_LOG}" \
  TEST_START_SEED=10000 N_EXP=5 \
  bash scripts/cluster_matched_paper_wave2_awr.sh
echo "=== square Wave1+Wave2 ALL DONE $(date -Iseconds) ===" | tee -a "${LOG}"
