#!/usr/bin/env bash
# Shared config for the RoboTwin pipeline. Edit ONCE, then run 00..09 in order.
# All other scripts `source` this file.
set -euo pipefail

# --- paths ---
export OAT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # the oat/ project dir
export ROBOTWIN_DIR="${ROBOTWIN_DIR:-$HOME/RoboTwin}"              # where you cloned RoboTwin
export DEVICE="${DEVICE:-cuda:0}"

# --- task / data ---
export TASK="dual_bottles_pick_easy"          # RoboTwin task id (verify with 02_inspect)
export TASK_CFG="dual_bottles_pick"           # the config filename under config/task/*/robotwin/
export NDEMO=500                              # demos to use
export SRC_DIR="${OAT_DIR}/data/robotwin_src"           # where the LeRobot dataset lands (01_download)
export DATA_FORMAT="lerobot"                  # lerobot (recommended, unified) | hdf5 (native collect)
export HF_REPO="lerobot/robotwin_unified"     # HF dataset id (LeRobot v3.0, 79.6GB, all 50 tasks)

# --- outputs ---
export ZARR="${OAT_DIR}/data/robotwin/${TASK}_N${NDEMO}.zarr"
export TOK_CKPT="${OAT_DIR}/my_models/tokenizer_robotwin_${TASK}.ckpt"   # set from 05 output
export POLICY_CKPT="${OAT_DIR}/my_models/policy_robotwin_${TASK}.ckpt"   # set from 06 output
export AWR_CKPT="${OAT_DIR}/my_models/policy_robotwin_${TASK}_awr.ckpt"

# --- render backend (SAPIEN has its own; keep egl for any mujoco bits) ---
export MUJOCO_GL=egl
cd "${OAT_DIR}"
echo "[config] TASK=${TASK} NDEMO=${NDEMO} OAT_DIR=${OAT_DIR}"
