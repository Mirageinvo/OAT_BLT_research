#!/usr/bin/env bash
# After full MG download: slim → delete full → convert → next task.
# Usage: watch_g0_chain.sh close_drawer [next_task ...]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
export PATH="${HOME}/.local/bin:/Library/Frameworks/Python.framework/Versions/3.12/bin:${PATH}"
PYTHON="${PYTHON:-$(command -v python3)}"
SEED="${SEED:-0}"

slim_and_convert() {
  local TASK_SNAKE="$1"
  local PASCAL
  case "${TASK_SNAKE}" in
    close_drawer) PASCAL=CloseDrawer ;;
    coffee_press_button) PASCAL=CoffeePressButton ;;
    turn_off_microwave) PASCAL=TurnOffMicrowave ;;
    turn_off_sink_faucet) PASCAL=TurnOffSinkFaucet ;;
    *) echo "bad task"; exit 2 ;;
  esac
  local HDF5_ROOT="${ROOT}/data/robocasa/hdf5"
  local HUMAN="${HDF5_ROOT}/${PASCAL}/human/demo_gentex_im128_randcams.hdf5"
  local MG_FULL="${HDF5_ROOT}/${PASCAL}/mg/demo_gentex_im128_randcams.hdf5"
  local MG_SLIM="${HDF5_ROOT}/${PASCAL}/mg/demo_gentex_im128_randcams_M150_s${SEED}.hdf5"
  local ZARR="${ROOT}/data/robocasa/${TASK_SNAKE}_N200.zarr"
  local LOG="${ROOT}/logs/g0_${TASK_SNAKE}.log"

  echo "[$(date -u +%H:%M:%S)] wait MG full for ${TASK_SNAKE} ..." | tee -a "${LOG}"
  # wait until file stable & > 20GB (full mg ~24.3G)
  LAST_SZ=0
  while true; do
    PARTIAL="${MG_FULL}.partial"
    CURL_ALIVE=0
    pgrep -f "${PASCAL}/mg/demo_gentex_im128_randcams.hdf5" >/dev/null 2>&1 && CURL_ALIVE=1

    if [[ -f "${MG_FULL}" ]]; then
      SZ=$(stat -f%z "${MG_FULL}" 2>/dev/null || echo 0)
      if (( CURL_ALIVE == 0 && SZ > 20000000000 )); then
        echo "[$(date -u +%H:%M:%S)] MG ready $(du -h "${MG_FULL}" | awk '{print $1}')" | tee -a "${LOG}"
        break
      fi
      if (( CURL_ALIVE == 0 && SZ <= 20000000000 )); then
        echo "[$(date -u +%H:%M:%S)] MG incomplete (${SZ} B) — restart download" | tee -a "${LOG}"
        rm -f "${MG_FULL}" "${PARTIAL}"
        bash scripts/download_robocasa_paper_hdf5.sh "${PASCAL}/mg" >>"${LOG}" 2>&1 &
        sleep 30
      fi
    elif [[ -f "${PARTIAL}" ]]; then
      SZ=$(stat -f%z "${PARTIAL}" 2>/dev/null || echo 0)
      if (( SZ != LAST_SZ )); then
        echo "[$(date -u +%H:%M:%S)] downloading… $(du -h "${PARTIAL}" | awk '{print $1}')" | tee -a "${LOG}"
        LAST_SZ=$SZ
      fi
      if (( CURL_ALIVE == 0 )); then
        echo "[$(date -u +%H:%M:%S)] curl died on partial — resume" | tee -a "${LOG}"
        bash scripts/download_robocasa_paper_hdf5.sh "${PASCAL}/mg" >>"${LOG}" 2>&1 &
        sleep 30
      fi
    else
      if (( CURL_ALIVE == 0 )); then
        echo "[$(date -u +%H:%M:%S)] start MG download" | tee -a "${LOG}"
        bash scripts/download_robocasa_paper_hdf5.sh "${PASCAL}/mg" >>"${LOG}" 2>&1 &
        sleep 30
      fi
    fi
    sleep 60
  done

  if [[ ! -f "${MG_SLIM}" ]]; then
    echo "[$(date -u +%H:%M:%S)] slim M150 ..." | tee -a "${LOG}"
    "${PYTHON}" scripts/slim_robocasa_mg_hdf5.py \
      --src "${MG_FULL}" --dst "${MG_SLIM}" --n 150 --seed "${SEED}" \
      2>&1 | tee -a "${LOG}"
  fi

  echo "[$(date -u +%H:%M:%S)] rm full MG" | tee -a "${LOG}"
  rm -f "${MG_FULL}"

  FORCE=()
  [[ -d "${ZARR}" ]] && FORCE=(--force)
  echo "[$(date -u +%H:%M:%S)] convert N200 ..." | tee -a "${LOG}"
  "${PYTHON}" scripts/convert_robocasa_dataset.py \
    --task "${TASK_SNAKE}" \
    --human-hdf5 "${HUMAN}" \
    --mg-hdf5 "${MG_SLIM}" \
    --n-human 50 --n-machine 150 --seed "${SEED}" \
    "${FORCE[@]}" 2>&1 | tee -a "${LOG}"

  echo "[$(date -u +%H:%M:%S)] DONE ${TASK_SNAKE}" | tee -a "${LOG}"
  df -h "${ROOT}" | tail -1 | tee -a "${LOG}"
}

download_mg() {
  local TASK_SNAKE="$1"
  local PASCAL
  case "${TASK_SNAKE}" in
    close_drawer) PASCAL=CloseDrawer ;;
    coffee_press_button) PASCAL=CoffeePressButton ;;
    turn_off_microwave) PASCAL=TurnOffMicrowave ;;
    turn_off_sink_faucet) PASCAL=TurnOffSinkFaucet ;;
  esac
  bash scripts/download_robocasa_paper_hdf5.sh "${PASCAL}/mg"
}

TASKS=("$@")
if [[ ${#TASKS[@]} -eq 0 ]]; then
  TASKS=(close_drawer coffee_press_button turn_off_microwave turn_off_sink_faucet)
fi

# First task: MG download already started separately for close_drawer.
FIRST="${TASKS[0]}"
slim_and_convert "${FIRST}"

for ((i=1; i<${#TASKS[@]}; i++)); do
  T="${TASKS[$i]}"
  echo "[$(date -u +%H:%M:%S)] download MG for ${T}"
  download_mg "${T}"
  slim_and_convert "${T}"
done

echo "[$(date -u +%H:%M:%S)] ALL TASKS DONE — validate"
"${PYTHON}" scripts/validate_robocasa_data.py
