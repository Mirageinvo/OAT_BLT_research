#!/usr/bin/env bash
# Replan probes on BASE TopK ckpts (independent of BoN/AWR Wave1/2).
# Protocol: OAT8, -n 1, n_test=10, n_parallel_envs=1 (same as Can/MW probes).
#
# Order (serial — host RAM tight with RC coffee_press):
#   1) Lift ep-0900 (run A)
#   2) coffee-pull ep-1000
#   3) Lift ep-1400 (run B / paper TopK)
set -euo pipefail
cd /workspace/oat
source .venv/bin/activate
export MUJOCO_GL=egl
export OAT_USE_UV_RUN=0
export PYTHONUNBUFFERED=1
GPU="${GPU:-1}"
# shellcheck disable=SC1091
source scripts/cluster_gpu_env.sh "${GPU}"
bash scripts/patch_robosuite_egl_assert.sh
mkdir -p logs

run_one() {
  local name="$1" ckpt="$2" out="$3" log="$4"
  shift 4
  local extra=("$@")
  {
    echo "=== replan ${name} START $(date -Iseconds) ==="
    echo "ckpt=${ckpt}"
    echo "protocol: OAT8 -n 1 --n_test 10 --n_parallel_envs 1 gpu=${GPU}"
  } | tee "${log}"
  rm -rf "${out}"
  # Do NOT mkdir OUT — eval_policy_sim prompts Overwrite if dir exists.
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID}" MUJOCO_GL=egl \
    python scripts/eval_policy_sim.py \
      -c "${ckpt}" -o "${out}" -d "${OAT_DEVICE}" -n 1 \
      --use_k_tokens 8 --entropy_threshold 0 \
      --n_test 10 --n_parallel_envs 1 \
      "${extra[@]}" \
      2>&1 | tee -a "${log}"
  echo "=== DONE replan ${name} $(date -Iseconds) ===" | tee -a "${log}"
}

run_one lift900 \
  output/20260719/144024_train_oatpolicy_lift_N200/checkpoints/ep-0900_sr-0.930.ckpt \
  eval_out/replan_probe_lift900 \
  logs/replan_probe_lift900.log

run_one coffee1000 \
  output/20260720/090816_train_oatpolicy_mw-coffee-pull_st_N50/checkpoints/ep-1000_sr-0.432.ckpt \
  eval_out/replan_probe_coffee-pull \
  logs/replan_probe_coffee-pull.log \
  --env_task_name coffee-pull

run_one lift1400 \
  output/20260719/144024_train_oatpolicy_lift_N200/checkpoints/ep-1400_sr-0.950.ckpt \
  eval_out/replan_probe_lift1400 \
  logs/replan_probe_lift1400.log

echo "=== ALL replan chain DONE $(date -Iseconds) ==="
