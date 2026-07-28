#!/usr/bin/env bash
# LIBERO policy-forward latency: baseline (single) + BoN N=8/16/32, NO AWR.
# Adapted from cluster_latency_paper_done.sh for the LIBERO suite + BoN N-sweep.
#
# Everything is env-overridable — set what differs on your cluster, defaults match
# AGENT_GUIDE_TABLE_C_PRIME_LATENCY.md. Nothing is measured until you eyeball the
# "CONFIG" block it prints.
#
# Usage (inside paper docker):
#   bash scripts/cluster_latency_libero.sh
#   GPU=1 BASE_CKPT=my_models/policy_ep-0250_sr-0.596.ckpt bash scripts/cluster_latency_libero.sh
#   FAIR_KV=0 bash scripts/cluster_latency_libero.sh          # -> main Table C (deployed, no KV single)
set -euo pipefail

# ----------------------------- CONFIG (override via env) ----------------------
WORKDIR="${WORKDIR:-/workspace/my_project/OAT_BLT_research/oat}"   # repo workdir inside docker
VENV="${VENV:-.venv}"                         # venv to activate (NOT .venv_robocasa)
GIT_REPO="${GIT_REPO:-/workspace/my_project/OAT_BLT_research}"     # checkout WITH .git (repo root)
GPU="${GPU:-0}"
BASE_CKPT="${BASE_CKPT:-my_models/policy_ep-0250_sr-0.596.ckpt}"   # LIBERO base policy
SUITE="${SUITE:-libero10}"                    # label -> output dir (obs come from ckpt cfg)
BON_NS="${BON_NS:-8,16,32}"                   # BoN N values to time
REPS="${REPS:-10}"
TRIALS="${TRIALS:-8}"
WARMUP="${WARMUP:-20}"
N_OBS="${N_OBS:-50}"                          # val obs batches cycled during timing (task variety)
FAIR_KV="${FAIR_KV:-1}"                        # 1 = apples-to-apples KV (Table C'); 0 = deployed (Table C)
# ------------------------------------------------------------------------------

cd "${WORKDIR}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
export MUJOCO_GL=egl
export OAT_USE_UV_RUN=0

# shellcheck disable=SC1091
source scripts/cluster_gpu_env.sh "${GPU}"     # sets CUDA_VISIBLE_DEVICES + OAT_DEVICE
# robosuite EGL patch only matters for live rendering; latency uses dataset obs (no sim render),
# and the stock patch hardcodes cd /workspace/oat → run best-effort, never abort.
bash scripts/patch_robosuite_egl_assert.sh 2>/dev/null \
  || echo "(robosuite EGL patch skipped — not needed for dataset-obs latency)"

# git provenance — required by measure_latency_paper.py (docker has no .git).
# Derive from GIT_REPO unless already exported.
if [[ -z "${OAT_GIT_COMMIT:-}" ]]; then
  if git -C "${GIT_REPO}" rev-parse HEAD >/dev/null 2>&1; then
    OAT_GIT_COMMIT="$(git -C "${GIT_REPO}" rev-parse HEAD)"
    OAT_GIT_BRANCH="$(git -C "${GIT_REPO}" rev-parse --abbrev-ref HEAD)"
    OAT_GIT_DIRTY="$(git -C "${GIT_REPO}" status --porcelain | grep -q . && echo 1 || echo 0)"
  else
    echo "ERROR: no git at GIT_REPO=${GIT_REPO}. Set GIT_REPO=<host checkout> or export OAT_GIT_COMMIT." >&2
    exit 1
  fi
fi
export OAT_GIT_COMMIT OAT_GIT_BRANCH="${OAT_GIT_BRANCH:-unknown}" OAT_GIT_DIRTY="${OAT_GIT_DIRTY:-0}"

# fair-KV vs deployed
EXTRA_FLAGS=(--skip_awr --include_single --bon_ns "${BON_NS}")
if [[ "${FAIR_KV}" == "1" ]]; then
  EXTRA_FLAGS+=(--fair_kv)
  OUT_JSON="output/eval/matched_s10000/${SUITE}/latency_fair_kv_n16.json"
  LABEL="fair-KV"
else
  OUT_JSON="output/eval/matched_s10000/${SUITE}/latency.json"
  LABEL="deployed"
fi
mkdir -p logs
LOG="logs/latency_libero_${LABEL}_gpu${GPU}.log"

# sanity: base ckpt present
if [[ ! -f "${BASE_CKPT}" ]]; then
  echo "ERROR: base ckpt not found: ${WORKDIR}/${BASE_CKPT}" >&2
  exit 1
fi

{
  echo "=== CONFIG (check before it runs) ==="
  echo "  workdir=${WORKDIR}  venv=${VENV}  gpu=${CUDA_VISIBLE_DEVICES}  device=${OAT_DEVICE}"
  echo "  suite=${SUITE}  base_ckpt=${BASE_CKPT}"
  echo "  modes=single + bon(${BON_NS})  AWR=skipped  protocol=${LABEL}"
  echo "  reps=${REPS} trials=${TRIALS} warmup=${WARMUP} n_obs=${N_OBS} batch=1"
  echo "  git=${OAT_GIT_COMMIT} (${OAT_GIT_BRANCH}) dirty=${OAT_GIT_DIRTY}"
  echo "  out=${OUT_JSON}"
  echo "===================================="
} | tee "${LOG}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl \
  OAT_GIT_COMMIT="${OAT_GIT_COMMIT}" OAT_GIT_BRANCH="${OAT_GIT_BRANCH}" OAT_GIT_DIRTY="${OAT_GIT_DIRTY}" \
  python scripts/measure_latency_paper.py \
    --suite "${SUITE}" \
    --base_ckpt "${BASE_CKPT}" \
    -d "${OAT_DEVICE}" \
    --reps "${REPS}" --trials "${TRIALS}" --warmup "${WARMUP}" --n_obs "${N_OBS}" \
    "${EXTRA_FLAGS[@]}" \
    2>&1 | tee -a "${LOG}"

echo "" | tee -a "${LOG}"
echo "=== SUMMARY (${OUT_JSON}) ===" | tee -a "${LOG}"
python3 - "${OUT_JSON}" <<'PY' 2>&1 | tee -a "${LOG}"
import json, sys
d = json.load(open(sys.argv[1]))
m = d["modes"]
print(f"{'mode':8} {'ms (mean of trial medians)':>28}   {'Δ vs single':>12}")
base = m.get("single", {}).get("median_ms")
for k in ("single", "bon", "bon16", "bon32"):
    if k not in m: continue
    v = m[k]
    dm = f"+{v['median_ms']-base:.2f}" if (base is not None and k!="single") else ""
    print(f"{k:8} {v['median_ms']:>10.2f} ± {v['std_ms']:.2f}            {dm:>12}")
print(f"\ngpu={d.get('gpu_name')}  torch={d.get('torch_version')}  git={d.get('git_commit','')[:10]}")
PY
echo "=== DONE $(date -Iseconds) ===" | tee -a "${LOG}"
