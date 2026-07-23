#!/usr/bin/env bash
# One-shot: attach plateau→literal5 watchers to live RoboCasa trains + queue dead ones.
# Run inside docker. Creates tmux sessions; does not kill existing policy trains.
#
#   bash scripts/_launch_rc_baseline_watchers.sh
set -euo pipefail
cd /workspace/oat
mkdir -p logs my_models

launch_watch() {
  local name="$1" task="$2" run_dir="$3" train_tmux="$4" gpu="$5"
  if tmux has-session -t "${name}" 2>/dev/null; then
    echo "[skip] tmux ${name} already exists"
    return 0
  fi
  tmux new-session -d -s "${name}" \
    "cd /workspace/oat && \
     TASK=${task} RUN_DIR=${run_dir} TRAIN_TMUX=${train_tmux} GPU=${gpu} \
     MIN_EPOCH=${MIN_EPOCH:-2000} N_BELOW=${N_BELOW:-4} KILL_TRAIN=${KILL_TRAIN:-0} \
     TRAIN_END_EPOCH=${TRAIN_END_EPOCH:-4500} \
     bash scripts/_launch_rc_plateau_to_literal5.sh \
     2>&1 | tee logs/${name}.log"
  echo "[ok] started ${name} → ${task} (watch ${train_tmux})"
}

# Live trains
launch_watch rc_watch_close close_drawer \
  output/20260723/041317_train_oatpolicy_close_drawer_N200 rc_close 0
launch_watch rc_watch_sink turn_off_sink_faucet \
  output/20260723/041317_train_oatpolicy_turn_off_sink_faucet_N200 rc_sink 1

# Coffee: dead (SIGKILL) — resume when RAM frees, then watcher
if ! tmux has-session -t rc_coffee 2>/dev/null; then
  tmux new-session -d -s rc_coffee \
    "cd /workspace/oat && NEED_MIB=${COFFEE_NEED_MIB:-20000} GPU=${COFFEE_GPU:-1} \
     bash scripts/_launch_rc_coffee_resume.sh"
  echo "[ok] started rc_coffee (resume when RAM free)"
fi
if ! tmux has-session -t rc_watch_coffee 2>/dev/null; then
  tmux new-session -d -s rc_watch_coffee \
    "cd /workspace/oat && \
     TASK=coffee_press_button \
     RUN_DIR=output/20260721/204916_train_oatpolicy_coffee_press_button_N200 \
     TRAIN_TMUX=rc_coffee GPU=${COFFEE_GPU:-1} \
     MIN_EPOCH=${MIN_EPOCH:-2000} N_BELOW=${N_BELOW:-4} KILL_TRAIN=${KILL_TRAIN:-0} \
     TRAIN_END_EPOCH=${TRAIN_END_EPOCH:-4500} \
     bash scripts/_launch_rc_plateau_to_literal5.sh \
     2>&1 | tee logs/rc_watch_coffee.log"
  echo "[ok] started rc_watch_coffee"
fi

# Microwave: OOM early — queue train when free; watcher waits for run dir + tmux
if ! tmux has-session -t rc_microwave 2>/dev/null; then
  tmux new-session -d -s rc_microwave \
    "cd /workspace/oat && NEED_MIB=${MW_NEED_MIB:-20000} GPU=${MW_GPU:-0} \
     bash scripts/_launch_rc_microwave_when_free.sh"
  echo "[ok] started rc_microwave (train when RAM free)"
fi
# Microwave run dir may change on fresh start — watcher resolves latest dir
if ! tmux has-session -t rc_watch_microwave 2>/dev/null; then
  tmux new-session -d -s rc_watch_microwave \
    "cd /workspace/oat && \
     while true; do
       d=\$(ls -dt output/*/train_oatpolicy_turn_off_microwave_N200 2>/dev/null | head -1 || true)
       if [[ -n \"\$d\" && -d \"\$d/checkpoints\" ]] && tmux has-session -t rc_microwave 2>/dev/null; then
         # wait until at least one TopK OR train past epoch 0 eval started writing
         if ls \"\$d\"/checkpoints/ep-*_sr-*.ckpt >/dev/null 2>&1 || [[ -f \"\$d/logs.json\" ]]; then
           echo \"[watch_mw] attaching to \$d\"
           TASK=turn_off_microwave RUN_DIR=\"\$d\" TRAIN_TMUX=rc_microwave GPU=${MW_GPU:-0} \
             MIN_EPOCH=${MIN_EPOCH:-2000} N_BELOW=${N_BELOW:-4} KILL_TRAIN=${KILL_TRAIN:-0} \
             TRAIN_END_EPOCH=${TRAIN_END_EPOCH:-4500} \
             bash scripts/_launch_rc_plateau_to_literal5.sh
           break
         fi
       fi
       echo \"[watch_mw] waiting for microwave train/ckpts \$(date -Iseconds)\"
       sleep 90
     done 2>&1 | tee logs/rc_watch_microwave.log"
  echo "[ok] started rc_watch_microwave"
fi

tmux ls
echo "=== RC baseline watchers launched $(date -Iseconds) ==="
