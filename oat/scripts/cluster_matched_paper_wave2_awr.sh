#!/usr/bin/env bash
# Paper Wave 2 — fresh AWR for one suite that finished Wave 1 at matched_s10000.
#
# Anti-leak (RESOLUTIONPLAN):
#   collect --seed 0   (NOT 1000 selection, NOT 10000 report)
#   eval    TEST_START_SEED=10000 via cluster_matched_triplet SKIP_BASELINE_BON=1
#
# Artifact names (do not reuse exploratory policy_awr_* / awr_*.npz):
#   my_datasets/awr_s10000_<suite>.npz
#   my_models/awr_s10000_<suite>.ckpt
#   logs/awr_s10000_<suite>_wave2_gpu{G}.log
#   output/eval/matched_s10000/<suite>/awr_n5/   (eval)
#
# Usage:
#   SUITE=box-close GPU=0 bash scripts/cluster_matched_paper_wave2_awr.sh
#   SUITE=disassemble GPU=1 bash scripts/cluster_matched_paper_wave2_awr.sh
set -euo pipefail
cd /workspace/oat
source .venv/bin/activate
export MUJOCO_GL=egl
export OAT_USE_UV_RUN=0

SUITE="${SUITE:?Set SUITE=box-close|disassemble|...}"
GPU="${GPU:-0}"
# shellcheck disable=SC1091
source scripts/cluster_gpu_env.sh "${GPU}"
bash scripts/patch_robosuite_egl_assert.sh

TEST_START_SEED="${TEST_START_SEED:-10000}"
COLLECT_SEED="${COLLECT_SEED:-0}"          # anti-leak collect pool (not 1000 / not report)
TRAIN_SEED="${TRAIN_SEED:-0}"              # train_awr shuffle; same family as COLLECT_SEED; ≠ Δ
N_CHUNKS="${N_CHUNKS:-20000}"
BON_N="${BON_N:-8}"
N_WORKERS="${N_WORKERS:-4}"
EPOCHS="${EPOCHS:-30}"
BETA="${BETA:-0.5}"
BETA_KL="${BETA_KL:-0.05}"
N_EXP="${N_EXP:-5}"

case "${SUITE}" in
  box-close)
    BASE_CKPT="${BASE_CKPT:-output/20260711/134439_train_oatpolicy_mw-box-close_st_N50/checkpoints/ep-2000_sr-0.552.ckpt}"
    ENV_TASK=box-close
    N_TASKS=1
    ;;
  disassemble)
    BASE_CKPT="${BASE_CKPT:-output/20260711/134440_train_oatpolicy_mw-disassemble_st_N50/checkpoints/ep-1400_sr-0.700.ckpt}"
    ENV_TASK=disassemble
    N_TASKS=1
    ;;
  coffee-pull)
    # Paper refit 20260720; override via BASE_CKPT= after TopK lock.
    BASE_CKPT="${BASE_CKPT:-output/20260720/090816_train_oatpolicy_mw-coffee-pull_st_N50/checkpoints/ep-1000_sr-0.432.ckpt}"
    ENV_TASK=coffee-pull
    N_TASKS=1
    ;;
  stick-pull)
    BASE_CKPT="${BASE_CKPT:-output/20260711/134439_train_oatpolicy_mw-stick-pull_st_N50/checkpoints/ep-0800_sr-0.212.ckpt}"
    ENV_TASK=stick-pull
    N_TASKS=1
    ;;
  can)
    BASE_CKPT="${BASE_CKPT:-output/20260706/173343_train_oatpolicy_can_N200/checkpoints/ep-1700_sr-0.940.ckpt}"
    ENV_TASK=""
    N_TASKS=1
    ;;
  lift)
    # Default ep-0900 = run A artifacts. Paper TopK lock = ep-1400 via BASE_CKPT / _launch_lift_ep1400_matched.sh.
    BASE_CKPT="${BASE_CKPT:-output/20260719/144024_train_oatpolicy_lift_N200/checkpoints/ep-0900_sr-0.930.ckpt}"
    ENV_TASK=""
    N_TASKS=1
    ;;
  square)
    # Live TopK lock via scripts/_launch_square_matched_on_plateau.sh (override BASE_CKPT).
    BASE_CKPT="${BASE_CKPT:-output/20260720/215024_train_oatpolicy_square_N200/checkpoints/ep-0700_sr-0.420.ckpt}"
    ENV_TASK=""
    N_TASKS=1
    ;;
  *)
    echo "Unknown SUITE=${SUITE}" >&2
    exit 1
    ;;
esac

WAVE1_ROOT="${WAVE1_ROOT:-output/eval/matched_s${TEST_START_SEED}/${SUITE}}"
[[ -f "${WAVE1_ROOT}/baseline_n${N_EXP}/eval_log.json" ]] || {
  echo "ERROR Wave1 baseline missing: ${WAVE1_ROOT}/baseline_n${N_EXP}/eval_log.json" >&2
  exit 1
}
[[ -f "${WAVE1_ROOT}/bon_n${BON_N}_n${N_EXP}/eval_log.json" ]] || {
  echo "ERROR Wave1 BoN missing: ${WAVE1_ROOT}/bon_n${BON_N}_n${N_EXP}/eval_log.json" >&2
  exit 1
}
[[ -f "${BASE_CKPT}" ]] || { echo "ERROR missing BASE_CKPT=${BASE_CKPT}" >&2; exit 1; }

# Paper naming — never exploratory policy_awr_* / awr_<suite>_bon.npz
# Override AWR_DS / AWR_CKPT for dual-base runs (e.g. lift ep0900 vs ep1400).
AWR_DS="${AWR_DS:-my_datasets/awr_s${TEST_START_SEED}_${SUITE}.npz}"
AWR_CKPT="${AWR_CKPT:-my_models/awr_s${TEST_START_SEED}_${SUITE}.ckpt}"
LOG="${LOG:-logs/awr_s${TEST_START_SEED}_${SUITE}_wave2_gpu${GPU}.log}"
mkdir -p my_datasets my_models logs

{
  echo "=== PAPER WAVE2 AWR ${SUITE} $(date -Iseconds) ==="
  echo "base_ckpt=${BASE_CKPT}"
  echo "collect_seed=${COLLECT_SEED}  (anti-leak: not 1000, not ${TEST_START_SEED})"
  echo "train_seed=${TRAIN_SEED}  (DataLoader only; not env / not Δ)"
  echo "eval_seed=${TEST_START_SEED}"
  echo "awr_ds=${AWR_DS}"
  echo "awr_ckpt=${AWR_CKPT}"
  echo "wave1_root=${WAVE1_ROOT}"
  echo "gpu=${CUDA_VISIBLE_DEVICES} device=${OAT_DEVICE}"
} | tee "${LOG}"

echo "" | tee -a "${LOG}"
echo "[STEP1] collect AWR dataset (BoN-distill N=${BON_N}, seed=${COLLECT_SEED})" | tee -a "${LOG}"
if [[ -f "${AWR_DS}" && "${FORCE_RECOLLECT:-0}" != "1" ]]; then
  echo "[skip] collect — ${AWR_DS} already exists (set FORCE_RECOLLECT=1 to redo)" | tee -a "${LOG}"
else
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl \
    python scripts/collect_awr_dataset.py \
      -c "${BASE_CKPT}" \
      -o "${AWR_DS}" \
      -d "${OAT_DEVICE}" \
      --n_chunks "${N_CHUNKS}" \
      --bon_n "${BON_N}" \
      --n_tasks "${N_TASKS}" \
      --seed "${COLLECT_SEED}" \
      --n_workers "${N_WORKERS}" \
      2>&1 | tee -a "${LOG}"
fi

echo "" | tee -a "${LOG}"
echo "[STEP2] validate ${AWR_DS}" | tee -a "${LOG}"
python scripts/validate_awr.py -i "${AWR_DS}" 2>&1 | tee -a "${LOG}"

echo "" | tee -a "${LOG}"
echo "[STEP3] train AWR -> ${AWR_CKPT}" | tee -a "${LOG}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  python scripts/train_awr.py \
    -i "${AWR_DS}" \
    -c "${BASE_CKPT}" \
    -o "${AWR_CKPT}" \
    --beta "${BETA}" \
    --beta_kl "${BETA_KL}" \
    --epochs "${EPOCHS}" \
    --ordering uniform \
    --seed "${TRAIN_SEED}" \
    2>&1 | tee -a "${LOG}"

echo "" | tee -a "${LOG}"
echo "[STEP4] paper eval AWR @ seed ${TEST_START_SEED}; keep Wave1 baseline/BoN" | tee -a "${LOG}"
# Separate eval log; never truncate Wave1 matched_s10000 logs.
# Override EVAL_LOG for dual-base runs (e.g. lift_ep1400) so we don't clobber suite default.
EVAL_LOG="${EVAL_LOG:-logs/awr_s${TEST_START_SEED}_${SUITE}_wave2_eval_gpu${GPU}.log}"
SUITE="${SUITE}" GPU="${GPU}" \
  BASE_CKPT="${BASE_CKPT}" \
  AWR_CKPT="${AWR_CKPT}" \
  LOG="${EVAL_LOG}" \
  OUT_ROOT="${WAVE1_ROOT}" \
  SKIP_AWR=0 SKIP_BASELINE_BON=1 FORCE_RERUN=0 \
  TEST_START_SEED="${TEST_START_SEED}" N_EXP="${N_EXP}" \
  bash scripts/cluster_matched_triplet.sh 2>&1 | tee -a "${LOG}"

echo "=== WAVE2 ALL DONE ${SUITE} $(date -Iseconds) ===" | tee -a "${LOG}"
echo "artifacts: ds=${AWR_DS} ckpt=${AWR_CKPT} eval=${WAVE1_ROOT}/awr_n${N_EXP}/ summary=${WAVE1_ROOT}/summary.json"
