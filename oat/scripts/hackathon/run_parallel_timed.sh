#!/usr/bin/env bash
# Parallel hackathon run:
#   1) train lift/can/square simultaneously for TIME_LIMIT_SEC (default 2.5h)
#   2) pick latest checkpoint per task
#   3) quick eval + 1 success demo per task (all 3 GPUs in parallel)
#
# Needs 3 GPUs (default TRAIN_GPUS=0,1,2). Single-GPU: set TRAIN_GPUS=0,0,0 at OOM risk.
set -euo pipefail
cd "$(dirname "$0")/../.."

export MUJOCO_GL="${MUJOCO_GL:-egl}"

TIME_LIMIT_SEC="${TIME_LIMIT_SEC:-9000}"   # 2h 30m
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2}"
EVAL_GPUS="${EVAL_GPUS:-${TRAIN_GPUS}}"
STAGE="${STAGE:-hackathon_assets}"
OUT="${OUT:-hackathon_output}"
POL_DIR="${OUT}/policies/timed"
PROG="${OUT}/progress"
LOG_DIR="${OUT}/logs"

IFS=',' read -r -a TRAIN_GPU_ARR <<< "${TRAIN_GPUS}"
IFS=',' read -r -a EVAL_GPU_ARR <<< "${EVAL_GPUS}"

if [[ ! -f "${STAGE}/tokenizers/lift.ckpt" ]]; then
  echo "Run first: bash scripts/hackathon/00_download_assets.sh"
  exit 1
fi

mkdir -p "${POL_DIR}" "${PROG}" "${LOG_DIR}"

pick_run_dir() {
  local task="$1"
  ls -td output/*/*train_oatpolicy_${task}_* 2>/dev/null | head -1 || true
}

pick_latest_ckpt() {
  local run_dir="$1"
  local ckpt=""
  ckpt="$(ls -t "${run_dir}"/checkpoints/ep-*.ckpt 2>/dev/null | head -1 || true)"
  if [[ -n "${ckpt}" ]]; then
    echo "${ckpt}"
    return
  fi
  if [[ -f "${run_dir}/checkpoints/latest.ckpt" ]]; then
    echo "${run_dir}/checkpoints/latest.ckpt"
  fi
}

echo "============================================================"
echo " PHASE 1 — parallel train ${TIME_LIMIT_SEC}s (~$(( TIME_LIMIT_SEC / 3600 ))h $(( (TIME_LIMIT_SEC % 3600) / 60 ))m)"
echo " GPUs: ${TRAIN_GPUS}"
echo " Started: $(date -Iseconds)"
echo "============================================================"

train_pids=()
i=0
for task in lift can square; do
  gpu="${TRAIN_GPU_ARR[$i]:-${TRAIN_GPU_ARR[0]}}"
  echo "  launch train ${task} on GPU ${gpu}"
  TIME_LIMIT_SEC="${TIME_LIMIT_SEC}" GPU="${gpu}" \
    bash scripts/hackathon/train_one_timed.sh "${task}" "${gpu}" &
  train_pids+=($!)
  i=$((i + 1))
done

fail=0
for pid in "${train_pids[@]}"; do
  wait "${pid}" || fail=1
done
echo "All trainers finished (some may have hit timeout). wait_fail=${fail}"

echo ""
echo "============================================================"
echo " PHASE 2 — collect latest checkpoints"
echo "============================================================"

declare -A TASK_CKPT
for task in lift can square; do
  run_dir="$(pick_run_dir "${task}")"
  if [[ -z "${run_dir}" ]]; then
    echo "ERROR: no run dir for ${task}"
    exit 1
  fi
  ckpt="$(pick_latest_ckpt "${run_dir}")"
  if [[ -z "${ckpt}" || ! -f "${ckpt}" ]]; then
    echo "ERROR: no checkpoint for ${task} in ${run_dir}"
    exit 1
  fi
  dest="${POL_DIR}/${task}_latest.ckpt"
  cp "${ckpt}" "${dest}"
  TASK_CKPT["${task}"]="${dest}"
  echo "  ${task}: ${ckpt} -> ${dest}"
done

manifest="${OUT}/timed_run_manifest.json"
python3 - <<PY
import json, pathlib, subprocess

out = pathlib.Path("${OUT}")
tasks = {}
for task, ckpt in [
    ("lift", "${TASK_CKPT[lift]:-}"),
    ("can", "${TASK_CKPT[can]:-}"),
    ("square", "${TASK_CKPT[square]:-}"),
]:
    run_dir = subprocess.check_output(
        f"ls -td output/*/*train_oatpolicy_{task}_* 2>/dev/null | head -1",
        shell=True, text=True,
    ).strip()
    tasks[task] = {"run_dir": run_dir, "ckpt": ckpt, "copied": str(out / "policies/timed" / f"{task}_latest.ckpt")}
pathlib.Path("${manifest}").write_text(json.dumps({
    "time_limit_sec": int("${TIME_LIMIT_SEC}"),
    "tasks": tasks,
}, indent=2))
PY

echo ""
echo "============================================================"
echo " PHASE 2b — training progress plots"
echo "============================================================"
for task in lift can square; do
  run_dir="$(pick_run_dir "${task}")"
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
if command -v uv >/dev/null 2>&1; then
  uv run python scripts/hackathon/plot_training_progress_combined.py \
    --progress_dir "${PROG}" -o "${PROG}/all_tasks_training.png" 2>/dev/null || true
else
  python3 scripts/hackathon/plot_training_progress_combined.py \
    --progress_dir "${PROG}" -o "${PROG}/all_tasks_training.png" 2>/dev/null || true
fi

echo ""
echo "============================================================"
echo " PHASE 3 — parallel quick eval + demo gifs"
echo " GPUs: ${EVAL_GPUS}"
echo " Started: $(date -Iseconds)"
echo "============================================================"

eval_pids=()
i=0
for task in lift can square; do
  gpu="${EVAL_GPU_ARR[$i]:-${EVAL_GPU_ARR[0]}}"
  ckpt="${TASK_CKPT[$task]}"
  echo "  launch eval+demo ${task} on GPU ${gpu}"
  OUT="${OUT}" GPU="${gpu}" bash scripts/hackathon/eval_and_demo_one.sh "${task}" "${ckpt}" "${gpu}" &
  eval_pids+=($!)
  i=$((i + 1))
done

for pid in "${eval_pids[@]}"; do
  wait "${pid}" || fail=1
done

echo ""
echo "============================================================"
echo " DONE $(date -Iseconds)"
echo " Policies:  ${POL_DIR}/"
echo " Progress:  ${PROG}/"
echo " Eval:      ${OUT}/eval/timed/*/eval_log.json"
echo " Demos:     ${OUT}/gifs/*_success_*.gif"
echo " Manifest:  ${manifest}"
echo "============================================================"
