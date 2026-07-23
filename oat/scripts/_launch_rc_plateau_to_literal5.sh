#!/usr/bin/env bash
# RoboCasa absolute baseline: wait for a *mature* TopK → lock → literal-5 Wave1.
#
# Conservative by default (do NOT kill half-trained fits):
#   - MIN_EPOCH=2000  (rollout_every=100 → ≥20 train-evals before any lock)
#   - N_BELOW=4       (need 4 consecutive evals strictly below the peak)
#   - KILL_TRAIN=0    (lock + Wave1; leave train running unless explicitly set)
#   - dead tmux alone never locks (resume / OOM) — need FINISHED log or
#     best_ep >= TRAIN_END_EPOCH (default 4500)
#
# Usage (cluster docker):
#   TASK=close_drawer RUN_DIR=output/.../train_oatpolicy_close_drawer_N200 \
#     TRAIN_TMUX=rc_close GPU=0 bash scripts/_launch_rc_plateau_to_literal5.sh
#
# Optional: SKIP_BON=1  KILL_TRAIN=1  N_BELOW=4  MIN_EPOCH=2000  POLL_SEC=60
set -euo pipefail
cd /workspace/oat

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export OAT_USE_UV_RUN=0
export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"

TASK="${TASK:?set TASK=close_drawer|coffee_press_button|turn_off_microwave|turn_off_sink_faucet}"
case "${TASK}" in
  close_drawer|coffee_press_button|turn_off_microwave|turn_off_sink_faucet) ;;
  *) echo "ERROR: unknown TASK=${TASK}"; exit 2 ;;
esac

RUN_DIR="${RUN_DIR:?set RUN_DIR=output/.../train_oatpolicy_${TASK}_N200}"
CKPT_DIR="${RUN_DIR}/checkpoints"
TRAIN_TMUX="${TRAIN_TMUX:?set TRAIN_TMUX=...}"
GPU="${GPU:-0}"
N_BELOW="${N_BELOW:-4}"
MIN_EPOCH="${MIN_EPOCH:-2000}"
TRAIN_END_EPOCH="${TRAIN_END_EPOCH:-4500}"  # natural-end / late-dead lock bar
KILL_TRAIN="${KILL_TRAIN:-0}"
POLL_SEC="${POLL_SEC:-60}"
SKIP_BON="${SKIP_BON:-0}"
FORCE_RERUN="${FORCE_RERUN:-1}"
LOCK_FILE="${LOCK_FILE:-my_models/robocasa_${TASK}_topk_lock.txt}"
LOG="${LOG:-logs/rc_plateau_${TASK}_gpu${GPU}.log}"
# Policy train logs (resume uses *_resume.log)
TRAIN_LOG_GLOB="${TRAIN_LOG_GLOB:-logs/train_policy_robocasa_${TASK}_s0*.log}"
mkdir -p logs my_models

export CKPT_DIR N_BELOW MIN_EPOCH

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
  python3 - <<'PY'
import re, os
from pathlib import Path
ckpt_dir = Path(os.environ["CKPT_DIR"])
n_below_need = int(os.environ["N_BELOW"])
min_epoch = int(os.environ["MIN_EPOCH"])
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
if best_ep < min_epoch:
    print(f"WAITING\t{best_sr}\t{best_ep}\t0\t{last_ep}\t{last_sr}")
    raise SystemExit(0)
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

train_finished_cleanly() {
  # True if any matching train log says the policy run finished (not SIGKILL).
  local f
  for f in ${TRAIN_LOG_GLOB}; do
    [[ -f "${f}" ]] || continue
    if grep -q "robocasa policy finished" "${f}" 2>/dev/null; then
      return 0
    fi
    if grep -q "=== robocasa policy finished" "${f}" 2>/dev/null; then
      return 0
    fi
  done
  return 1
}

eval_active() {
  tmux has-session -t "${TRAIN_TMUX}" 2>/dev/null || return 1
  local pane
  pane="$(tmux capture-pane -t "${TRAIN_TMUX}" -p 2>/dev/null | tr '\r' '\n' | tail -50)"
  echo "${pane}" | grep -qE "Eval oatpolicy_rgb in RoboCasa::${TASK} [0-9]+/"
}

ckpt_fingerprint() {
  ls -1 "${CKPT_DIR}"/ep-*_sr-*.ckpt 2>/dev/null | sort | md5sum | awk '{print $1}'
}

lock_and_wave1() {
  local reason="$1"
  AFTER="$(best_topk)"
  BASE_CKPT="$(echo "${AFTER}" | cut -f1)"
  AFTER_SR="$(echo "${AFTER}" | cut -f2)"
  AFTER_EP="$(echo "${AFTER}" | cut -f3)"
  {
    echo "[lock] reason=${reason}"
    echo "BASE_CKPT=${BASE_CKPT}"
    echo "sr=${AFTER_SR} ep=${AFTER_EP}"
  } | tee -a "${LOG}"
  [[ -f "${BASE_CKPT}" ]] || { echo "ERROR missing ${BASE_CKPT}"; exit 1; }

  {
    echo "base_ckpt=${BASE_CKPT}"
    echo "sr=${AFTER_SR}"
    echo "ep=${AFTER_EP}"
    echo "run_dir=${RUN_DIR}"
    echo "task=${TASK}"
    echo "reason=${reason}"
    echo "locked_at=$(date -Iseconds)"
    echo "rule=peak_then_${N_BELOW}_below_min_ep_${MIN_EPOCH}_kill_${KILL_TRAIN}"
  } | tee "${LOCK_FILE}" | tee -a "${LOG}"

  if [[ "${KILL_TRAIN}" == "1" ]] && tmux has-session -t "${TRAIN_TMUX}" 2>/dev/null; then
    echo "[stop] KILL_TRAIN=1 — killing ${TRAIN_TMUX} $(date -Iseconds)" | tee -a "${LOG}"
    tmux kill-session -t "${TRAIN_TMUX}" || true
    sleep 5
  else
    echo "[keep] train left running (KILL_TRAIN=${KILL_TRAIN}) $(date -Iseconds)" | tee -a "${LOG}"
  fi

  echo "=== Wave1 literal-5 START $(date -Iseconds) ===" | tee -a "${LOG}"
  SUITE="${TASK}" BASE_CKPT="${BASE_CKPT}" GPU="${GPU}" \
    SKIP_BON="${SKIP_BON}" FORCE_RERUN="${FORCE_RERUN}" \
    bash scripts/cluster_robocasa_literal5_wave1.sh
  echo "=== RC ${TASK} plateau→literal5 ALL DONE $(date -Iseconds) ===" | tee -a "${LOG}"
  echo "summary: output/eval/matched_s10000/robocasa/${TASK}/summary_literal5.json" | tee -a "${LOG}"
}

{
  echo "=== RC plateau→literal5 START $(date -Iseconds) ==="
  echo "task=${TASK} run_dir=${RUN_DIR}"
  echo "train_tmux=${TRAIN_TMUX} n_below=${N_BELOW} min_epoch=${MIN_EPOCH} train_end_epoch=${TRAIN_END_EPOCH}"
  echo "kill_train=${KILL_TRAIN} poll=${POLL_SEC}s gpu=${GPU} skip_bon=${SKIP_BON}"
  if [[ -d "${CKPT_DIR}" ]] && ls "${CKPT_DIR}"/ep-*_sr-*.ckpt >/dev/null 2>&1; then
    echo "topk_now=$(best_topk)"
  else
    echo "topk_now=(none yet)"
  fi
} | tee "${LOG}"

prev_fp="$(ckpt_fingerprint || true)"
while true; do
  has_ckpts=0
  if [[ -d "${CKPT_DIR}" ]] && ls "${CKPT_DIR}"/ep-*_sr-*.ckpt >/dev/null 2>&1; then
    has_ckpts=1
  fi

  if ! tmux has-session -t "${TRAIN_TMUX}" 2>/dev/null; then
    if [[ "${has_ckpts}" -eq 1 ]]; then
      status_line="$(plateau_status)"
      best_ep="$(echo "${status_line}" | cut -f3)"
      if train_finished_cleanly; then
        echo "[done] train finished cleanly — lock Top-1 $(date -Iseconds)" | tee -a "${LOG}"
        lock_and_wave1 "train_finished"
        exit 0
      fi
      if [[ "${best_ep}" -ge "${TRAIN_END_EPOCH}" ]]; then
        echo "[done] train dead but best_ep=${best_ep}>=${TRAIN_END_EPOCH} — lock Top-1 $(date -Iseconds)" | tee -a "${LOG}"
        lock_and_wave1 "late_dead_ep${best_ep}"
        exit 0
      fi
      echo "[wait] train tmux gone early (best_ep=${best_ep} < ${TRAIN_END_EPOCH}, no finished marker) — waiting resume $(date -Iseconds)" | tee -a "${LOG}"
    else
      echo "[wait] no ckpts yet — waiting for train $(date -Iseconds)" | tee -a "${LOG}"
    fi
    sleep "${POLL_SEC}"
    continue
  fi

  if eval_active; then
    pane_tail="$(tmux capture-pane -t "${TRAIN_TMUX}" -p 2>/dev/null | tr '\r' '\n' | grep -E 'Eval oatpolicy_rgb|Training epoch' | tail -2 || true)"
    echo "[wait] train-eval active $(date -Iseconds) ${pane_tail}" | tee -a "${LOG}"
    sleep "${POLL_SEC}"
    continue
  fi
  sleep 20
  fp="$(ckpt_fingerprint || true)"
  if [[ -n "${fp}" && "${fp}" != "${prev_fp}" ]]; then
    echo "[ckpt] set changed ${prev_fp} → ${fp} $(date -Iseconds)" | tee -a "${LOG}"
    prev_fp="${fp}"
    for _ in 1 2 3 4 5; do
      sleep 10
      fp2="$(ckpt_fingerprint || true)"
      [[ "${fp2}" == "${fp}" ]] && break
      fp="${fp2}"
    done
    prev_fp="${fp}"
  fi

  if [[ "${has_ckpts}" -eq 0 ]]; then
    echo "[wait] no TopK files yet $(date -Iseconds)" | tee -a "${LOG}"
    sleep "${POLL_SEC}"
    continue
  fi

  status_line="$(plateau_status)"
  echo "[status] ${status_line} $(date -Iseconds)" | tee -a "${LOG}"
  status="$(echo "${status_line}" | cut -f1)"
  if [[ "${status}" == "PLATEAU" ]]; then
    echo "[plateau] mature peak + ${N_BELOW} below (min_ep=${MIN_EPOCH}) — lock $(date -Iseconds)" | tee -a "${LOG}"
    lock_and_wave1 "plateau"
    exit 0
  fi
  sleep "${POLL_SEC}"
done
