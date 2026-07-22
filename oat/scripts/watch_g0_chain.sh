#!/usr/bin/env bash
# After full MG download: slim → delete full → convert → next task.
# Usage: watch_g0_chain.sh close_drawer [next_task ...]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
# shellcheck source=robocasa_hdf5_lib.sh
source "${ROOT}/scripts/robocasa_hdf5_lib.sh"
export PATH="${HOME}/.local/bin:/Library/Frameworks/Python.framework/Versions/3.12/bin:${PATH}"
PYTHON="${PYTHON:-}"
if [[ -z "${PYTHON}" ]]; then
  for c in \
    /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
    "${ROOT}/.venv/bin/python" \
    "$(command -v python3)"; do
    if [[ -n "${c}" && -x "${c}" ]] && "${c}" -c "import numpy,zarr" 2>/dev/null; then
      PYTHON="${c}"
      break
    fi
  done
fi
: "${PYTHON:?no python with numpy+zarr}"
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

  local MG_EXPECTED
  MG_EXPECTED="$(robocasa_resolve_expected_bytes "${PASCAL}" mg)" \
    || { echo "[err] no expected MG size for ${PASCAL}" | tee -a "${LOG}"; exit 1; }
  echo "[$(date -u +%H:%M:%S)] wait MG for ${TASK_SNAKE} (expected $(robocasa_format_bytes "${MG_EXPECTED}") = ${MG_EXPECTED} B) ..." \
    | tee -a "${LOG}"
  LAST_SZ=0
  while true; do
    PARTIAL="${MG_FULL}.partial"
    CURL_ALIVE=0
    pgrep -f "${PASCAL}/mg/demo_gentex_im128_randcams.hdf5" >/dev/null 2>&1 && CURL_ALIVE=1

    # Pick the largest local artifact (partial may be ahead of a stale final).
    SZ=0
    [[ -f "${MG_FULL}" ]] && SZ=$(stat -f%z "${MG_FULL}" 2>/dev/null || echo 0)
    if [[ -f "${PARTIAL}" ]]; then
      PSZ=$(stat -f%z "${PARTIAL}" 2>/dev/null || echo 0)
      (( PSZ > SZ )) && SZ=$PSZ
    fi

    if robocasa_bytes_complete "${SZ}" "${MG_EXPECTED}"; then
      if (( CURL_ALIVE == 0 )); then
        if [[ ! -f "${MG_FULL}" ]] && [[ -f "${PARTIAL}" ]]; then
          echo "[$(date -u +%H:%M:%S)] finalize partial → final" | tee -a "${LOG}"
          bash scripts/download_robocasa_paper_hdf5.sh "${PASCAL}/mg" >>"${LOG}" 2>&1
        elif [[ -f "${MG_FULL}" ]] && "${PYTHON}" "${ROOT}/scripts/verify_robocasa_hdf5.py" "${MG_FULL}" >>"${LOG}" 2>&1; then
          :
        elif [[ -f "${MG_FULL}" ]]; then
          echo "[$(date -u +%H:%M:%S)] MG size OK but HDF5 bad — re-download" | tee -a "${LOG}"
          rm -f "${MG_FULL}" "${PARTIAL}"
          bash scripts/download_robocasa_paper_hdf5.sh "${PASCAL}/mg" >>"${LOG}" 2>&1 &
          sleep 30
          continue
        fi
        echo "[$(date -u +%H:%M:%S)] MG ready $(du -h "${MG_FULL}" | awk '{print $1}') (${SZ}/${MG_EXPECTED} B)" | tee -a "${LOG}"
        break
      fi
    elif (( SZ != LAST_SZ )); then
      local pct=$(( SZ * 100 / MG_EXPECTED ))
      echo "[$(date -u +%H:%M:%S)] downloading… $(robocasa_format_bytes "${SZ}") / $(robocasa_format_bytes "${MG_EXPECTED}") (~${pct}%)" | tee -a "${LOG}"
      LAST_SZ=$SZ
    fi

    if (( CURL_ALIVE == 0 )); then
      if [[ -f "${PARTIAL}" ]] || [[ -f "${MG_FULL}" ]]; then
        echo "[$(date -u +%H:%M:%S)] curl idle — resume download (no wipe)" | tee -a "${LOG}"
      else
        echo "[$(date -u +%H:%M:%S)] start MG download" | tee -a "${LOG}"
      fi
      bash scripts/download_robocasa_paper_hdf5.sh "${PASCAL}/mg" >>"${LOG}" 2>&1 &
      sleep 30
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

  # validate then drop slim (zarr is the paper artifact)
  echo "[$(date -u +%H:%M:%S)] validate ${TASK_SNAKE} ..." | tee -a "${LOG}"
  "${PYTHON}" scripts/validate_robocasa_data.py --task "${TASK_SNAKE}" 2>&1 | tee -a "${LOG}"
  echo "[$(date -u +%H:%M:%S)] rm slim M150" | tee -a "${LOG}"
  rm -f "${MG_SLIM}"

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
