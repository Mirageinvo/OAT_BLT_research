#!/usr/bin/env bash
# Hackathon POC — 2 tasks (lift + can), 1 GPU, time-split train + quick eval + success clip.
#
#   Total budget default 2.5h → ~1.25h lift, ~1.25h can (sequential, no OOM).
#   Then n_test=10 SR + 1 success mp4/gif per task.
#
# Usage:
#   cd oat && MUJOCO_GL=egl bash scripts/hackathon/run_hackathon_poc.sh
#
set -euo pipefail
cd "$(dirname "$0")/../.."

export MUJOCO_GL="${MUJOCO_GL:-egl}"

TOTAL_SEC="${TOTAL_SEC:-9000}"       # 2h 30m total
GPU="${GPU:-0}"
TASKS=(lift can)
STAGE="${STAGE:-hackathon_assets}"
OUT="${OUT:-hackathon_output}"
POL_DIR="${OUT}/policies/poc"
PROG="${OUT}/progress"
LOG_DIR="${OUT}/logs"

N_TEST="${N_TEST:-10}"
BATCH="${BATCH:-48}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-15}"
ROLLOUT_EVERY="${ROLLOUT_EVERY:-250}"
HOLD="${HOLD_AFTER_SUCCESS:-80}"

PER_TASK_SEC=$(( TOTAL_SEC / ${#TASKS[@]} ))

if [[ ! -f "${STAGE}/tokenizers/lift.ckpt" ]]; then
  echo "Run: bash scripts/hackathon/00_download_assets.sh"
  exit 1
fi

mkdir -p "${POL_DIR}" "${PROG}" "${LOG_DIR}"

pick_run_dir() {
  local task="$1"
  ls -td output/*/*train_oatpolicy_${task}_* 2>/dev/null | head -1 || true
}

pick_latest_ckpt() {
  local run_dir="$1"
  local ckpt
  ckpt="$(ls -t "${run_dir}"/checkpoints/ep-*.ckpt 2>/dev/null | head -1 || true)"
  [[ -n "${ckpt}" ]] && { echo "${ckpt}"; return; }
  [[ -f "${run_dir}/checkpoints/latest.ckpt" ]] && echo "${run_dir}/checkpoints/latest.ckpt"
}

echo "============================================================"
echo " HACKATHON POC — ${TASKS[*]} on GPU ${GPU}"
echo " Total ${TOTAL_SEC}s (~$(( TOTAL_SEC / 3600 ))h $(( (TOTAL_SEC % 3600) / 60 ))m)"
echo " Per task ${PER_TASK_SEC}s (~$(( PER_TASK_SEC / 60 )) min)"
echo " Started $(date -Iseconds)"
echo "============================================================"

declare -A TASK_CKPT

for task in "${TASKS[@]}"; do
  echo ""
  echo ">>> TRAIN ${task} (${PER_TASK_SEC}s) <<<"
  TIME_LIMIT_SEC="${PER_TASK_SEC}" BATCH="${BATCH}" \
    CHECKPOINT_EVERY="${CHECKPOINT_EVERY}" ROLLOUT_EVERY="${ROLLOUT_EVERY}" \
    bash scripts/hackathon/train_one_timed.sh "${task}" "${GPU}"

  run_dir="$(pick_run_dir "${task}")"
  ckpt="$(pick_latest_ckpt "${run_dir}")"
  if [[ -z "${ckpt}" || ! -f "${ckpt}" ]]; then
    echo "ERROR: no ckpt for ${task}"
    exit 1
  fi
  dest="${POL_DIR}/${task}_latest.ckpt"
  cp "${ckpt}" "${dest}"
  TASK_CKPT["${task}"]="${dest}"
  echo "  ckpt: ${dest}"

  if [[ -f "${run_dir}/logs.json" ]]; then
    if command -v uv >/dev/null 2>&1; then
      uv run python scripts/hackathon/plot_training_progress.py \
        --run_dir "${run_dir}" --task "${task}" --output_dir "${PROG}" || true
    else
      python3 scripts/hackathon/plot_training_progress.py \
        --run_dir "${run_dir}" --task "${task}" --output_dir "${PROG}" || true
    fi
  fi
done

echo ""
echo "============================================================"
echo " EVAL + DEMO (${N_TEST} ep SR + 1 success clip per task)"
echo "============================================================"

for task in "${TASKS[@]}"; do
  echo ">>> EVAL ${task} <<<"
  OUT="${OUT}" GPU="${GPU}" N_TEST="${N_TEST}" HOLD_AFTER_SUCCESS="${HOLD}" \
    bash scripts/hackathon/eval_and_demo_one.sh "${task}" "${TASK_CKPT[$task]}" "${GPU}"
done

echo ""
echo "============================================================"
echo " DONE $(date -Iseconds)"
echo "  policies: ${POL_DIR}/"
echo "  SR logs:  ${OUT}/eval/timed/{lift,can}/eval_log.json"
echo "  gifs:     ${OUT}/gifs/{lift,can}_success_*.gif"
echo "  plots:    ${PROG}/{lift,can}_training_curve.png"
echo "============================================================"
