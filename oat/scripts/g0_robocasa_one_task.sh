#!/usr/bin/env bash
# G0 pipeline: download → slim MG(150) → convert N200 zarr → free full MG.
# One task at a time (full mg_im ≈ 24GB). Protocol: oat/ROBOCASA.md
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
export PATH="${HOME}/.local/bin:${ROOT}/.venv/bin:${PATH}"
PYTHON="${ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="$(command -v python3)"
fi
UV="$(command -v uv || true)"
run_py() {
  if [[ -n "${UV}" ]]; then
    uv run python "$@"
  else
    "${PYTHON}" "$@"
  fi
}

TASK_SNAKE="${1:?usage: $0 <close_drawer|coffee_press_button|turn_off_microwave|turn_off_sink_faucet>}"
SEED="${SEED:-0}"
KEEP_FULL_MG="${KEEP_FULL_MG:-0}"

case "${TASK_SNAKE}" in
  close_drawer) PASCAL=CloseDrawer ;;
  coffee_press_button) PASCAL=CoffeePressButton ;;
  turn_off_microwave) PASCAL=TurnOffMicrowave ;;
  turn_off_sink_faucet) PASCAL=TurnOffSinkFaucet ;;
  *) echo "unknown task ${TASK_SNAKE}"; exit 2 ;;
esac

HDF5_ROOT="${ROOT}/data/robocasa/hdf5"
HUMAN="${HDF5_ROOT}/${PASCAL}/human/demo_gentex_im128_randcams.hdf5"
MG_FULL="${HDF5_ROOT}/${PASCAL}/mg/demo_gentex_im128_randcams.hdf5"
MG_SLIM="${HDF5_ROOT}/${PASCAL}/mg/demo_gentex_im128_randcams_M150_s${SEED}.hdf5"
ZARR="${ROOT}/data/robocasa/${TASK_SNAKE}_N200.zarr"

echo "=== G0 ${TASK_SNAKE} (seed=${SEED}) ==="
df -h "${ROOT}" | tail -1

# 1) human + full mg
bash scripts/download_robocasa_paper_hdf5.sh "${PASCAL}/human"
bash scripts/download_robocasa_paper_hdf5.sh "${PASCAL}/mg"

# 2) slim MG to 150 demos
if [[ -f "${MG_SLIM}" ]]; then
  echo "[skip] slim exists $(du -h "${MG_SLIM}" | awk '{print $1}')"
else
  run_py scripts/slim_robocasa_mg_hdf5.py \
    --src "${MG_FULL}" --dst "${MG_SLIM}" --n 150 --seed "${SEED}"
fi

# 3) free ~24GB before convert if needed
if [[ "${KEEP_FULL_MG}" != "1" ]] && [[ -f "${MG_FULL}" ]]; then
  echo "[rm] full MG $(du -h "${MG_FULL}" | awk '{print $1}')"
  rm -f "${MG_FULL}"
fi

FORCE_FLAG=()
if [[ -d "${ZARR}" ]]; then
  FORCE_FLAG=(--force)
fi

run_py scripts/convert_robocasa_dataset.py \
  --task "${TASK_SNAKE}" \
  --hdf5-root "${HDF5_ROOT}" \
  --out-root "${ROOT}/data/robocasa" \
  --human-hdf5 "${HUMAN}" \
  --mg-hdf5 "${MG_SLIM}" \
  --n-human 50 --n-machine 150 --seed "${SEED}" \
  "${FORCE_FLAG[@]}"

echo "=== done ${TASK_SNAKE} → ${ZARR} ==="
df -h "${ROOT}" | tail -1
