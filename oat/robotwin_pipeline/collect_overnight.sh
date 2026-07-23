#!/usr/bin/env bash
# Overnight batch: collect several RoboTwin tasks sequentially, with DISK GUARD (the disk is 99%
# full and SHARED). Skips tasks already collected; deletes per-episode videos (not needed for
# training) to minimise footprint; continues to the next task on failure. Logs per task.
#
# Run (after the current pick_dual_bottles finishes), detached so it survives disconnect:
#   cd <oat>; nohup bash robotwin_pipeline/collect_overnight.sh > robotwin_pipeline/overnight.out 2>&1 &
#   tail -f robotwin_pipeline/overnight.out
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
set +e                                   # continue to next task even if one fails (override config's -e)

# --- edit this list (simple, likely-working RoboTwin tasks; avoid open_laptop = upstream bug) ---
TASKS=(
  pick_dual_bottles
  place_empty_cup
  stack_blocks_two
  beat_block_hammer
  handover_block
  click_bell
  lift_pot
  place_shoe
)
MIN_FREE_GB="${MIN_FREE_GB:-60}"         # STOP before the shared disk drops below this
DELETE_VIDEOS="${DELETE_VIDEOS:-1}"      # 1 = rm per-episode mp4 after each task (save disk)

cd "${ROBOTWIN_DIR}"
mkdir -p collect_logs
echo "[overnight] start $(date) | ROBOTWIN_DIR=${ROBOTWIN_DIR} | config=${ROBOTWIN_CONFIG} | floor=${MIN_FREE_GB}G"

for T in "${TASKS[@]}"; do
  avail=$(df -BG --output=avail . 2>/dev/null | tail -1 | tr -dc '0-9')
  if [ "${avail:-0}" -lt "${MIN_FREE_GB}" ]; then
    echo "[STOP] free ${avail}G < ${MIN_FREE_GB}G floor — aborting to protect the shared disk ($(date))"
    break
  fi
  outdir="data/${T}/${ROBOTWIN_CONFIG}"
  if ls "${outdir}"/*.hdf5 >/dev/null 2>&1; then
    echo "[skip] ${T} already has HDF5 in ${outdir}"
  else
    echo "[collect] ${T} | free=${avail}G | $(date)"
    python script/collect_data.py "${T}" "${ROBOTWIN_CONFIG}" > "collect_logs/${T}.log" 2>&1
    rc=$?
    echo "[done] ${T} rc=${rc} | $(date)"
  fi
  if [ "${DELETE_VIDEOS}" = "1" ] && [ -d "${outdir}/video" ]; then
    rm -rf "${outdir}/video" && echo "  (removed ${outdir}/video to save disk)"
  fi
done
echo "[overnight] finished $(date). HDF5 per task under ${ROBOTWIN_DIR}/data/<task>/${ROBOTWIN_CONFIG}/"
df -h "${ROBOTWIN_DIR}" | tail -1
