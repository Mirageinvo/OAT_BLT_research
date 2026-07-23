#!/usr/bin/env bash
# Square replan-count probe — same protocol as Can / Lift / MW:
#   OAT8, -n 1, --n_test 10, --n_parallel_envs 1, BASE TopK lock ckpt.
# Independent of BoN/AWR; safe anytime after TopK lock.
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

CKPT="${CKPT:-output/20260720/215024_train_oatpolicy_square_N200/checkpoints/ep-0700_sr-0.420.ckpt}"
OUT="${OUT:-eval_out/replan_probe_square}"
LOG="${LOG:-logs/replan_probe_square.log}"
[[ -f "${CKPT}" ]] || { echo "ERROR missing ${CKPT}"; exit 1; }

rm -rf "${OUT}"
mkdir -p logs
{
  echo "=== replan Square START $(date -Iseconds) ==="
  echo "ckpt=${CKPT}"
  echo "protocol: OAT8 -n 1 --n_test 10 --n_parallel_envs 1 gpu=${GPU}"
} | tee "${LOG}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID}" MUJOCO_GL=egl \
  python scripts/eval_policy_sim.py \
    -c "${CKPT}" -o "${OUT}" -d "${OAT_DEVICE}" -n 1 \
    --use_k_tokens 8 --entropy_threshold 0 \
    --n_test 10 --n_parallel_envs 1 \
    2>&1 | tee -a "${LOG}"

echo "=== DONE replan Square $(date -Iseconds) ===" | tee -a "${LOG}"
python3 - <<'PY' | tee -a "${LOG}"
import json
from pathlib import Path
p = Path("eval_out/replan_probe_square/eval_log.json")
d = json.loads(p.read_text())
keys = [
    "mean_replans_per_episode_mean",
    "std_replans_per_episode_mean",
    "max_replans_per_episode_mean",
    "mean_success_rate_mean",
]
print("--- extract ---")
for k in keys:
    print(f"{k}={d.get(k)}")
PY
