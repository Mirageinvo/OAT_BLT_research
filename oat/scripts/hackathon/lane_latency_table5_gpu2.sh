#!/usr/bin/env bash
# Table 5 selector-only latency queue — GPU2 only, one ckpt at a time.
# Canon (= can n32diag): warmup=50, 8 trials × 20 reps, Ns=8,16,32, batch 1,
# real candidates, oat_code_seed (LAST_CALL present). Skip can. Skip RoboCasa.
set -euo pipefail

GPU=2
CODE="${HOME}/oat_code_seed"
OUTDIR="${HOME}/oat_eval_out/latency"
LOG="${OUTDIR}/lane_gpu2_table5.log"
PY="${CODE}/.venv/bin/python"
GL="${HOME}/gl-prefix/usr/lib/x86_64-linux-gnu"
MUJOCO_BIN="${HOME}/.mujoco/mujoco210/bin"

mkdir -p "${OUTDIR}"
exec >>"${LOG}" 2>&1

gpu2_busy() {
  local apps extra p gpu
  apps=$(nvidia-smi -i "${GPU}" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -v '^$' | grep -v 'N/A' || true)
  extra=""
  for p in $(pgrep -f python || true); do
    gpu=$(tr '\0' '\n' < "/proc/${p}/environ" 2>/dev/null | grep '^CUDA_VISIBLE_DEVICES=' | cut -d= -f2 || true)
    if [[ "${gpu}" == "${GPU}" ]]; then
      extra="${extra} ${p}"
    fi
  done
  [[ -n "${apps}" || -n "${extra}" ]]
}

wait_empty() {
  local n=0
  while gpu2_busy; do
    echo "WAIT GPU${GPU} occupied $(date -Is) (n=${n})"
    n=$((n + 1))
    if (( n > 240 )); then
      echo "ABORT GPU${GPU} still occupied after 2h"
      exit 2
    fi
    sleep 30
  done
}

# paper Wave1 OAT8 ckpts; can already measured — skip
CKPTS=(
  "lift|${HOME}/aaai27_infra/models/checkpoints/selected_from_output/output/20260719/144024_train_oatpolicy_lift_N200/checkpoints/ep-1400_sr-0.950.ckpt"
  "square|${HOME}/oat_code/hackathon_assets/policies/_dl/policies/square/ep-0700_sr-0.420.ckpt"
  "box-close|${HOME}/aaai27_infra/models/checkpoints/selected_from_output/output/20260711/134439_train_oatpolicy_mw-box-close_st_N50/checkpoints/ep-2000_sr-0.552.ckpt"
  "coffee-pull|${HOME}/aaai27_infra/models/checkpoints/selected_from_output/output/20260720/090816_train_oatpolicy_mw-coffee-pull_st_N50/checkpoints/ep-1000_sr-0.432.ckpt"
  "stick-pull|${HOME}/aaai27_infra/models/checkpoints/selected_from_output/output/20260711/134439_train_oatpolicy_mw-stick-pull_st_N50/checkpoints/ep-0800_sr-0.212.ckpt"
  "disassemble|${HOME}/aaai27_infra/models/checkpoints/selected_from_output/output/20260711/134440_train_oatpolicy_mw-disassemble_st_N50/checkpoints/ep-1400_sr-0.700.ckpt"
)

echo "START_LANE gpu=${GPU} $(date -Is) host=$(hostname) code=${CODE}"
[[ -x "${PY}" ]] || { echo "ABORT no venv ${PY}"; exit 3; }
[[ -f "${CODE}/scripts/measure_selector_latency_gpu.py" ]] || { echo "ABORT no measure script"; exit 3; }
grep -q LAST_CALL "${CODE}/oat/policy/kdpe.py" || { echo "ABORT seed kdpe.py missing LAST_CALL"; exit 3; }

# hydra val datasets live under oat_code/data; seed tree must see them
if [[ ! -e "${CODE}/data" && -d "${HOME}/oat_code/data" ]]; then
  ln -s "${HOME}/oat_code/data" "${CODE}/data"
  echo "linked ${CODE}/data -> ${HOME}/oat_code/data"
fi

export PATH="${HOME}/.local/bin:${PATH}"
export OAT_USE_UV_RUN=0 PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export MUJOCO_PY_MUJOCO_PATH="${HOME}/.mujoco/mujoco210"
export LD_LIBRARY_PATH="${GL}:${MUJOCO_BIN}:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MPLCONFIGDIR=/tmp/matplotlib-askhabaliev
export CUDA_VISIBLE_DEVICES="${GPU}"

cd "${CODE}"

for item in "${CKPTS[@]}"; do
  name="${item%%|*}"
  ckpt="${item#*|}"
  out="${OUTDIR}/selector_latency_gpu${GPU}_${name}"
  if [[ -f "${out}.json" ]]; then
    echo "SKIP ${name} already has ${out}.json"
    continue
  fi
  if [[ ! -f "${ckpt}" ]]; then
    echo "ABORT missing ckpt ${name} ${ckpt}"
    exit 4
  fi
  wait_empty
  if gpu2_busy; then
    echo "REFUSE ${name}: GPU${GPU} occupied at launch"
    exit 5
  fi
  echo "RUN ${name} gpu=${GPU} $(date -Is) ckpt=${ckpt}"
  "${PY}" scripts/measure_selector_latency_gpu.py \
    -c "${ckpt}" -d cuda:0 \
    --Ns 8,16,32 --trials 8 --reps 20 --warmup 50 --n_obs 8 \
    --use_k_tokens 8 --temperature 1.0 --topk 10 \
    --out "${out}"
  echo "DONE ${name} $(date -Is)"
  wait_empty
done

echo "LANE_COMPLETE $(date -Is)"
