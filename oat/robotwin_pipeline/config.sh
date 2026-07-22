#!/usr/bin/env bash
# Shared config for the RoboTwin pipeline. Edit ONCE, then run 00..09 in order.
# All other scripts `source` this file.
set -euo pipefail

# --- paths ---
export OAT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # the oat/ project dir
export ROBOTWIN_DIR="${ROBOTWIN_DIR:-$HOME/RoboTwin}"              # where you cloned RoboTwin
export DEVICE="${DEVICE:-cuda:0}"

# --- task / data ---
export TASK="pick_dual_bottles"               # RoboTwin env id (envs/pick_dual_bottles.py)
export TASK_CFG="dual_bottles_pick"           # the config filename under config/task/*/robotwin/
export ROBOTWIN_CONFIG="demo_clean"           # task_config for collect_data.py (demo_clean=Easy)
export NDEMO=50                               # demos (demo_clean episode_num; bump to 200 if underfit)
export SRC_DIR="${OAT_DIR}/data/robotwin_src/${TASK}"   # where generated HDF5 demos land
export DATA_FORMAT="hdf5"                     # hdf5 = generate locally with collect_data.py (small,
                                              # disk-safe). lerobot = 80GB unified download (AVOID on
                                              # the 99%-full shared disk).
export HF_REPO="lerobot/robotwin_unified"     # (only if you ever use --include one-task slice)
export N_COLLECT="${N_COLLECT:-200}"          # demos to GENERATE for the task (few GB)

# --- outputs ---
export ZARR="${OAT_DIR}/data/robotwin/${TASK}_N${NDEMO}.zarr"
export TOK_CKPT="${OAT_DIR}/my_models/tokenizer_robotwin_${TASK}.ckpt"   # set from 05 output
export POLICY_CKPT="${OAT_DIR}/my_models/policy_robotwin_${TASK}.ckpt"   # set from 06 output
export AWR_CKPT="${OAT_DIR}/my_models/policy_robotwin_${TASK}_awr.ckpt"

# --- render backend ---
export MUJOCO_GL=egl
# SAPIEN Vulkan ICD (fix for docker: points the loader at nvidia driver via SAPIEN's ICD).
# Needed because the container's /usr/share/vulkan/icd.d was empty; harmless if not.
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/home/docker_user/.local/lib/python3.11/site-packages/sapien/vulkan_library/nvidia_icd.json}"
cd "${OAT_DIR}"
echo "[config] TASK=${TASK} NDEMO=${NDEMO} OAT_DIR=${OAT_DIR}"
