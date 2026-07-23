#!/bin/bash
# Usage: bash eval.sh <task_name> <task_config> <oat_ckpt_path> <bon_n> <seed> <gpu_id>
#   e.g. bash eval.sh pick_dual_bottles demo_clean /path/policy.ckpt 1 0 0   # baseline
#        bash eval.sh pick_dual_bottles demo_clean /path/policy.ckpt 8 0 0   # BoN N=8
policy_name=OAT
task_name=${1}
task_config=${2}
oat_ckpt=${3}
bon_n=${4:-1}
seed=${5:-0}
gpu_id=${6:-0}
oat_dir=${OAT_DIR:-/workspace/my_project/OAT_BLT_research/oat}

export CUDA_VISIBLE_DEVICES=${gpu_id}
export PYTHONPATH=${oat_dir}:${PYTHONPATH}
export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/home/docker_user/.local/lib/python3.11/site-packages/sapien/vulkan_library/nvidia_icd.json}

cd ../..   # RoboTwin root
PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/OAT/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting default \
    --oat_ckpt ${oat_ckpt} \
    --bon_n ${bon_n} \
    --seed ${seed} \
    --policy_name ${policy_name}
