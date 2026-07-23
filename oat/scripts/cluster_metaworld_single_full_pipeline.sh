#!/usr/bin/env bash
# Full single-task specialist pipeline:
#   data -> tokenizer(top1) -> policy(full train eval) -> chain5(top-3 pick best) -> BoN/AWR.
#
# Required:
#   TASK=box-close|coffee-pull|disassemble|stick-pull
#
# Safe defaults:
# - does NOT kill existing tmux sessions
# - uses dedicated names/artifacts per task to avoid confusion
set -euo pipefail
cd /workspace/oat
source .venv/bin/activate

export OAT_USE_UV_RUN=0
export MUJOCO_GL=egl
GPU="${GPU:-0}"
TASK="${TASK:?Set TASK}"
SEED="${SEED:-0}"
NUM_DEMO="${NUM_DEMO:-50}"
DATE_TAG="$(date +%Y%m%d)"
TIME_TAG="$(date +%H%M%S)"
RUN_TAG="mwst_${TASK}_${DATE_TAG}_${TIME_TAG}"
LOG="${LOG:-logs/${RUN_TAG}_full_pipeline.log}"

mkdir -p logs output/eval my_datasets my_models
exec > >(tee -a "${LOG}") 2>&1

echo "=== [${RUN_TAG}] FULL single-task MetaWorld pipeline ==="
echo "task=${TASK} gpu=${GPU} seed=${SEED} num_demo=${NUM_DEMO}"

# 0) Ensure per-task dataset exists (split from mt4_N50.zarr if needed).
TASK_ZARR="data/metaworld/${TASK}_N${NUM_DEMO}.zarr"
if [[ ! -d "${TASK_ZARR}" ]]; then
  LOCK_DIR="data/metaworld/.mw_split_N${NUM_DEMO}.lock"
  echo "[DATA] Missing ${TASK_ZARR}; waiting lock ${LOCK_DIR}"
  while ! mkdir "${LOCK_DIR}" 2>/dev/null; do
    if [[ -d "${TASK_ZARR}" ]]; then
      break
    fi
    sleep 5
  done
  if [[ ! -d "${TASK_ZARR}" ]]; then
    echo "[DATA] Acquired lock; splitting from data/metaworld/mt4_N${NUM_DEMO}.zarr"
    python scripts/split_metaworld_mt4_zarr.py \
      --src "data/metaworld/mt4_N${NUM_DEMO}.zarr" \
      --task-name mt4 \
      --episodes-per-task "${NUM_DEMO}" \
      --out-root data/metaworld
  fi
  rmdir "${LOCK_DIR}" 2>/dev/null || true
fi
python scripts/validate_metaworld_data.py "${TASK_ZARR}" --episodes-per-task "${NUM_DEMO}" --num-tasks 1

# 1) Train tokenizer (top-1).
echo "[TOK] Train tokenizer"
TASK="${TASK}" GPU="${GPU}" SEED="${SEED}" NUM_DEMO="${NUM_DEMO}" \
  bash scripts/cluster_tokenizer_metaworld_single.sh

TOK_RUN_DIR="$(python - <<PY
import pathlib, re
task = "${TASK}"
root = pathlib.Path("output")
cands = []
for p in root.glob("**/*train_oattok_mw-*"):
    n = p.name
    if f"train_oattok_mw-{task}_st_N" in n:
        cands.append(p)
if not cands:
    raise SystemExit("No tokenizer run dir found")
cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
print(cands[0])
PY
)"
TOKENIZER_CKPT="$(python scripts/select_best_ckpt_by_name.py --run-dir "${TOK_RUN_DIR}" --metric mse --mode min --top 1)"
echo "[TOK] run_dir=${TOK_RUN_DIR}"
echo "[TOK] best=${TOKENIZER_CKPT}"

# 2) Train policy with full intermediate eval.
echo "[POL] Train policy (full eval)"
TASK="${TASK}" GPU="${GPU}" SEED="${SEED}" NUM_DEMO="${NUM_DEMO}" \
TOKENIZER_CKPT="${TOKENIZER_CKPT}" \
ROLLOUT_EVERY="${ROLLOUT_EVERY:-200}" N_TEST="${N_TEST:-250}" \
N_PARALLEL_ENVS="${N_PARALLEL_ENVS:-4}" NUM_WORKERS="${NUM_WORKERS:-8}" \
  bash scripts/cluster_policy_metaworld_single.sh

POL_RUN_DIR="$(python - <<PY
import pathlib
task = "${TASK}"
root = pathlib.Path("output")
cands = []
for p in root.glob("**/*train_oatpolicy_mw-*"):
    n = p.name
    if f"train_oatpolicy_mw-{task}_st_N" in n:
        cands.append(p)
if not cands:
    raise SystemExit("No policy run dir found")
cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
print(cands[0])
PY
)"
echo "[POL] run_dir=${POL_RUN_DIR}"

# 3) Chain5 eval on top-3 policy checkpoints; pick best by mean SR.
echo "[EVAL] chain5 over top-3 checkpoints"
TOP3="$(python scripts/select_best_ckpt_by_name.py --run-dir "${POL_RUN_DIR}" --metric sr --mode max --top 3)"
if [[ -z "${TOP3}" ]]; then
  echo "ERROR: no sr checkpoints in ${POL_RUN_DIR}/checkpoints"
  exit 1
fi

BEST_CKPT=""
BEST_MEAN="-1"
while IFS= read -r CKPT; do
  [[ -z "${CKPT}" ]] && continue
  CKPT_BASE="$(basename "${CKPT%.ckpt}")"
  OUT_ROOT="output/eval/metaworld_${TASK}_paper5_${CKPT_BASE}_${RUN_TAG}"
  TASK="${TASK}" CKPT="${CKPT}" GPU="${GPU}" OUT_ROOT="${OUT_ROOT}" \
    bash scripts/cluster_eval_metaworld_single_chain5.sh
  MEAN="$(python - <<PY
import json
with open("${OUT_ROOT}/summary.json") as f:
    s = json.load(f)
print(float(s["mean_success_rate"]))
PY
)"
  echo "[EVAL] ${CKPT_BASE} mean_sr=${MEAN}"
  if python - <<PY
import sys
curr=float("${MEAN}")
best=float("${BEST_MEAN}")
sys.exit(0 if curr>best else 1)
PY
  then
    BEST_MEAN="${MEAN}"
    BEST_CKPT="${CKPT}"
  fi
done <<< "${TOP3}"

if [[ -z "${BEST_CKPT}" ]]; then
  echo "ERROR: failed to choose best checkpoint."
  exit 1
fi
echo "[EVAL] selected best_ckpt=${BEST_CKPT} mean_sr=${BEST_MEAN}"

# 4) BoN -> AWR pipeline from selected best checkpoint.
echo "[BON/AWR] run pipeline from selected checkpoint"
TASK="${TASK}" CKPT="${BEST_CKPT}" GPU="${GPU}" NUM_DEMO="${NUM_DEMO}" \
LOG="logs/${RUN_TAG}_bon_awr.log" \
  bash scripts/cluster_metaworld_single_bon_awr.sh

echo "=== [${RUN_TAG}] DONE ==="
echo "task=${TASK}"
echo "tokenizer_run=${TOK_RUN_DIR}"
echo "policy_run=${POL_RUN_DIR}"
echo "best_policy_ckpt=${BEST_CKPT}"
