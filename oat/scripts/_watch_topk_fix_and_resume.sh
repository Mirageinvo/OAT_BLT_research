#!/usr/bin/env bash
# Wait until a RoboCasa policy train is NOT mid-sim-eval, then SIGINT + resume
# so TopKCheckpointManager disk-rebuild + k=2 takes effect.
#
# Usage (inside docker /workspace/oat):
#   TASK=coffee_press_button \
#     RUN_DIR=output/20260721/204916_train_oatpolicy_coffee_press_button_N200 \
#     TRAIN_TMUX=rc_coffee GPU=1 \
#     TOKENIZER_CKPT=output/.../ep-1940_mse-0.003.ckpt \
#     TRAIN_LOG=logs/train_policy_robocasa_coffee_press_button_s0_resume.log \
#     bash scripts/_watch_topk_fix_and_resume.sh
set -euo pipefail
cd /workspace/oat

TASK="${TASK:?}"
RUN_DIR="${RUN_DIR:?}"
TRAIN_TMUX="${TRAIN_TMUX:?}"
GPU="${GPU:-0}"
TOKENIZER_CKPT="${TOKENIZER_CKPT:?}"
TRAIN_LOG="${TRAIN_LOG:?}"
ROLLOUT_START_EPOCH="${ROLLOUT_START_EPOCH:-0}"
POLL_SEC="${POLL_SEC:-45}"
STABLE_SECS="${STABLE_SECS:-90}"   # must stay non-Eval this long before interrupt
LOG="${LOG:-logs/rc_topkfix_${TASK}.log}"
mkdir -p logs

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export OAT_USE_UV_RUN=0
export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"

VENV="${VENV:-/workspace/oat/.venv_robocasa}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

log() { echo "[topkfix ${TASK}] $* $(date -Iseconds)" | tee -a "${LOG}"; }

# 0 = mid-Eval (wait); 1 = safe (training/recon/idle)
log_is_safe() {
  python3 - "$TRAIN_LOG" <<'PY'
import sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file():
    raise SystemExit(0)  # no log yet → treat as safe/idle
raw = p.read_bytes()[-120_000:]
text = raw.decode("utf-8", errors="replace").replace("\r", "\n")
lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
if not lines:
    raise SystemExit(0)
# find last train-ish vs last eval-progress line
last_train = -1
last_eval = -1
for i, ln in enumerate(lines):
    if ln.startswith("Training epoch") or ln.startswith("Reconstruction epoch") or ln.startswith("Validation epoch"):
        last_train = i
    # live eval progress bars / episode headers
    if "Eval oatpolicy" in ln or ln.startswith("Eval "):
        last_eval = i
if last_eval > last_train:
    raise SystemExit(1)  # mid-eval
raise SystemExit(0)
PY
}

train_alive() {
  tmux has-session -t "${TRAIN_TMUX}" 2>/dev/null || return 1
  # python run_workspace still under that session?
  pgrep -af "run_workspace.py.*${TASK}" >/dev/null 2>&1
}

wait_until_safe() {
  local stable_for=0
  log "waiting until NOT mid-Eval (poll=${POLL_SEC}s, need stable ${STABLE_SECS}s)"
  while true; do
    if ! train_alive; then
      log "train not alive — will resume cold"
      return 0
    fi
    if log_is_safe; then
      stable_for=$((stable_for + POLL_SEC))
      log "safe window ${stable_for}/${STABLE_SECS}s (not mid-Eval)"
      if [[ "${stable_for}" -ge "${STABLE_SECS}" ]]; then
        return 0
      fi
    else
      stable_for=0
      log "mid-Eval — waiting"
    fi
    sleep "${POLL_SEC}"
  done
}

stop_train() {
  if ! tmux has-session -t "${TRAIN_TMUX}" 2>/dev/null; then
    log "tmux ${TRAIN_TMUX} gone"
    return 0
  fi
  log "sending C-c to ${TRAIN_TMUX}"
  tmux send-keys -t "${TRAIN_TMUX}" C-c
  # second C-c if stuck in tqdm
  sleep 5
  tmux send-keys -t "${TRAIN_TMUX}" C-c || true
  for i in $(seq 1 60); do
    if ! pgrep -af "run_workspace.py.*${TASK}" >/dev/null 2>&1; then
      log "train process exited"
      return 0
    fi
    sleep 2
  done
  log "WARN: still alive after C-c — escalate SIGTERM"
  pkill -TERM -f "run_workspace.py.*${TASK}" || true
  sleep 10
  pkill -KILL -f "run_workspace.py.*${TASK}" || true
}

resume_train() {
  # shellcheck disable=SC1091
  source scripts/cluster_gpu_env.sh "${GPU}"
  VENV="${VENV}" bash scripts/patch_robosuite_egl_assert.sh || true

  log "resume START gpu=${GPU} run=${RUN_DIR} topk.k=2"
  # Always recreate tmux session — send-keys into a dead pane is unreliable.
  tmux kill-session -t "${TRAIN_TMUX}" 2>/dev/null || true
  sleep 1
  tmux new-session -d -s "${TRAIN_TMUX}" bash -lc "
set -euo pipefail
cd /workspace/oat
source ${VENV}/bin/activate
source scripts/cluster_gpu_env.sh ${GPU}
export MUJOCO_GL=egl WANDB_MODE=disabled PYTHONUNBUFFERED=1 OAT_USE_UV_RUN=0
export LD_LIBRARY_PATH=\"\${HOME}/.mujoco/mujoco210/bin:\${LD_LIBRARY_PATH:-}\"
SKIP_G0B=1 HYDRA_FULL_ERROR=1 accelerate launch \
  --num_machines 1 --num_processes 1 \
  scripts/run_workspace.py \
  --config-name=train_oatpolicy \
  task/policy=robocasa/${TASK} \
  task.policy.lazy_eval=false \
  task.policy.env_runner.n_test=50 \
  task.policy.env_runner.n_parallel_envs=1 \
  task.policy.env_runner.test_start_seed=2000 \
  policy.action_tokenizer.checkpoint=${TOKENIZER_CKPT} \
  training.num_demo=200 \
  training.rollout_every=100 \
  training.rollout_start_epoch=${ROLLOUT_START_EPOCH} \
  training.seed=0 seed=0 \
  dataloader.num_workers=1 val_dataloader.num_workers=1 \
  checkpoint.topk.k=2 \
  checkpoint.topk.monitor_key=mean_success_rate \
  logging.mode=disabled \
  hydra.run.dir=${RUN_DIR} \
  training.resume=true \
  2>&1 | tee -a ${TRAIN_LOG}
"
  # wait up to ~2min for process (model load)
  for i in $(seq 1 24); do
    if pgrep -af "run_workspace.py.*${TASK}" >/dev/null 2>&1; then
      log "resume OK — process up"
      ls -lh "${RUN_DIR}/checkpoints/" 2>/dev/null | tee -a "${LOG}" || true
      return 0
    fi
    sleep 5
  done
  log "ERROR: resume process not found"
  tmux capture-pane -t "${TRAIN_TMUX}" -p 2>/dev/null | tail -40 | tee -a "${LOG}" || true
  return 1
}

log "=== watcher start ==="
wait_until_safe
stop_train
sleep 3
resume_train
log "=== watcher done ==="
