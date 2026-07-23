#!/usr/bin/env bash
# Sequential raw MH -> image HDF5 extract (can, then square).
# Patches env_args for the local robosuite stack, then runs dataset_states_to_obs.py.
#
# Usage (cluster docker, train untouched):
#   cd /workspace/oat && source .venv/bin/activate
#   export MUJOCO_GL=egl OAT_USE_UV_RUN=0
#   export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
#   nohup nice -n 19 bash scripts/extract_robomimic_mh_image.sh can square \
#     >> logs/extract_mh_image_pipeline.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

# demo_v15 raw HDF5 targets robosuite 1.5+ for state->image replay (model_file XML).
# Train/eval stays on robosuite 1.4 in .venv; extract uses an isolated env if present.
EXTRACT_VENV="${EXTRACT_VENV:-${ROOT}/.venv_extract}"
if [[ -f "${EXTRACT_VENV}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${EXTRACT_VENV}/bin/activate"
  echo "using extract venv: ${EXTRACT_VENV} (robosuite $(python -c 'import robosuite; print(robosuite.__version__)' 2>/dev/null || echo '?'))"
else
  # shellcheck disable=SC1091
  source "${ROOT}/.venv/bin/activate"
  echo "WARNING: ${EXTRACT_VENV} missing — using train .venv (robosuite 1.4 may fail on demo_v15)"
fi

ensure_robosuite_asset_compat() {
  local assets="${ROOT}/.venv/lib/python3.10/site-packages/robosuite/models/assets"
  if [[ -d "${assets}/mounts/meshes" && ! -e "${assets}/bases/meshes" ]]; then
    mkdir -p "${assets}/bases"
    ln -sfn ../mounts/meshes "${assets}/bases/meshes"
  fi
}
ensure_robosuite_asset_compat

TASKS=("$@")
if [[ ${#TASKS[@]} -eq 0 ]]; then
  TASKS=(can square)
fi

HDF5_ROOT="${HDF5_ROOT:-data/robomimic/hdf5_datasets}"
ROBOMIMIC_SCRIPTS="$(python -c 'import robomimic, os; print(os.path.join(os.path.dirname(robomimic.__file__), "scripts"))')"
NICE_LEVEL="${NICE_LEVEL:-19}"

robosuite_major_minor() {
  python -c 'import robosuite; v=robosuite.__version__.split("."); print(int(v[0]), int(v[1]))'
}

extract_one() {
  local task="$1"
  local raw="${HDF5_ROOT}/${task}/mh/demo_v15.hdf5"
  local out_dir="${HDF5_ROOT}/${task}/mh"
  local image_tmp="${out_dir}/image_v15.hdf5"
  local image_link="${HDF5_ROOT}/${task}_mh_image.hdf5"

  echo "========== $(date -Is) START ${task} =========="
  if [[ ! -f "${raw}" ]]; then
    echo "ERROR: missing raw ${raw}"
    exit 1
  fi
  if [[ -f "${image_link}" ]]; then
    echo "SKIP ${task}: ${image_link} already exists ($(du -h "${image_link}" | cut -f1))"
    return 0
  fi

  read -r rs_maj rs_min < <(robosuite_major_minor)
  if [[ "${rs_maj}" -eq 1 && "${rs_min}" -lt 5 ]]; then
    echo "[${task}] robosuite <1.5: patch env_args for reader stack (see patch_robomimic_env_args.py)"
    REF_HDF5="${REF_HDF5:-${HDF5_ROOT}/lift_mh_image.hdf5}"
    python scripts/patch_robomimic_env_args.py \
      --dataset "${raw}" \
      --reference "${REF_HDF5}"
  else
    echo "[${task}] robosuite 1.5+: use native demo_v15 env_args (no patch)"
  fi

  echo "[${task}] smoke: env metadata"
  python - <<PY
import json, h5py
from robomimic.utils.env_utils import create_env_for_data_processing
p = "${raw}"
e = json.loads(h5py.File(p, "r")["data"].attrs["env_args"])
env = create_env_for_data_processing(
    env_meta=e,
    camera_names=["agentview", "robot0_eye_in_hand"],
    camera_height=84, camera_width=84, reward_shaping=False,
)
print("  env_name=", e.get("env_name"), "robosuite_ok=", type(env).__name__)
PY

  rm -f "${image_tmp}"
  mkdir -p "${out_dir}" logs

  echo "[${task}] dataset_states_to_obs (nice -n ${NICE_LEVEL}) ..."
  (
    cd "${ROBOMIMIC_SCRIPTS}"
    nice -n "${NICE_LEVEL}" python dataset_states_to_obs.py --done_mode 2 \
      --dataset "${ROOT}/${raw}" \
      --output_name image_v15.hdf5 \
      --camera_names agentview robot0_eye_in_hand \
      --camera_height 84 --camera_width 84 \
      --compress --exclude-next-obs
  )

  if [[ ! -f "${image_tmp}" ]]; then
    echo "ERROR: extract did not create ${image_tmp}"
    exit 1
  fi

  cp -f "${image_tmp}" "${image_link}"
  echo "[${task}] done -> ${image_link} ($(du -h "${image_link}" | cut -f1))"
  echo "========== $(date -Is) END ${task} =========="
}

for task in "${TASKS[@]}"; do
  extract_one "${task}"
done

echo "All tasks finished: ${TASKS[*]}"
