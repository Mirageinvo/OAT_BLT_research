#!/usr/bin/env bash
# Parallel multi-task RoboTwin data collection.
# Each task writes to data/<task>/<CONFIG>/ -> SEPARATE folders, cannot mix.
# Edit TASKS below (verify names against `ls envs/`). Launches one background collector per task.
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
set +e   # don't abort the whole launch if one task is bad

# ---- EDIT THESE ----
TASKS=(pick_dual_bottles place_empty_cup beat_block_hammer handover_block)
CONFIG="${ROBOTWIN_CONFIG:-demo_clean}"     # Easy mode
NDEMO_COLLECT="${NDEMO_COLLECT:-500}"       # demos per task
# --------------------

cd "${ROBOTWIN_DIR}"
echo "[collect] RoboTwin=${ROBOTWIN_DIR}  config=${CONFIG}  demos/task=${NDEMO_COLLECT}"

# 1) set episode_num in the shared task_config (all tasks read the same value)
CFG_YML="${ROBOTWIN_DIR}/task_config/${CONFIG}.yml"
if [ -f "${CFG_YML}" ]; then
  sed -i "s/^ *episode_num:.*/  episode_num: ${NDEMO_COLLECT}/" "${CFG_YML}"
  echo "[collect] set episode_num=${NDEMO_COLLECT} in ${CFG_YML}"
  grep -n episode_num "${CFG_YML}"
else
  echo "[collect] WARN: ${CFG_YML} not found — set episode_num manually"
fi

LOGDIR="${HOME}/robotwin_collect_logs"; mkdir -p "${LOGDIR}"
echo "[collect] logs -> ${LOGDIR}"
launched=()
for T in "${TASKS[@]}"; do
  if [ ! -f "${ROBOTWIN_DIR}/envs/${T}.py" ]; then
    echo "[collect] SKIP '${T}' — envs/${T}.py does NOT exist. Available tasks:"
    ls "${ROBOTWIN_DIR}/envs/" | grep -v __ | sed 's/\.py$//' | xargs -n4 2>/dev/null || \
      ls "${ROBOTWIN_DIR}/envs/" | grep -v __ | sed 's/\.py$//'
    continue
  fi
  OUT="${ROBOTWIN_DIR}/data/${T}/${CONFIG}"
  LOG="${LOGDIR}/collect_${T}.log"
  nohup "${OATPY}" script/collect_data.py "${T}" "${CONFIG}" > "${LOG}" 2>&1 &
  echo "[collect] launched ${T}  (pid $!)  -> out=${OUT}  log=${LOG}"
  launched+=("${T}")
  sleep 3   # stagger starts so SAPIEN/GPU init doesn't collide
done

echo
echo "[collect] launched ${#launched[@]} task(s): ${launched[*]}"
echo "[collect] monitor:   tail -f ${LOGDIR}/collect_<task>.log"
echo "[collect] GPU/CPU:   watch -n2 nvidia-smi   |   htop"
echo "[collect] progress:  ls data/<task>/${CONFIG}/data/ | wc -l   (grows toward ${NDEMO_COLLECT})"
echo "[collect] jobs:      jobs -l    (running background collectors this shell)"
