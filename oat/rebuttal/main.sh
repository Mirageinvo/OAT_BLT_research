#!/usr/bin/env bash
# AAMAS-27 rebuttal main block (500 scenes, S3 settings, new scene protocol v2).
# Run from OAT_BLT_research/oat:
#   SEEDS=0,1,2 MAXJ=3 nohup bash rebuttal/main.sh > main_driver.log 2>&1 &
# SEEDS: generation seed list, fixed BEFORE looking at results. MAXJ: concurrent eval processes
# on the one GPU (SR unaffected; inference_ms from parallel runs is indicative only).
set -u
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-0}
PY=${PY:-.venv/bin/python}
SEEDS=${SEEDS:-0}
MAXJ=${MAXJ:-3}
CKPT=my_models/policy_ep-0250_sr-0.596.ckpt
OUT=eval_out/rebuttal_cs_ablation
COMMON="--use_k_tokens 8 --entropy_threshold 0 --temperature 1.0 --topk 10 --selector_seed 0 --n_parallel_envs 4"
S3="$COMMON --n_test 500 --test_start_seed 10000 --gen_seeds $SEEDS"
mkdir -p $OUT/main
log() { echo "[$(date '+%F %T')] $*"; }
log "SEEDS=$SEEDS MAXJ=$MAXJ commit=$(git rev-parse --short HEAD)"

# ---- 1. bandwidth calibration for cs_prefix_fixed: held-out seeds 20000+, frozen before eval ----
if [ ! -f $OUT/calib_h/h_calibration.json ]; then
  log "start calib_h"
  $PY scripts/eval_policy_sim.py -c $CKPT -o $OUT/calib_h --force $COMMON \
    --n_test 40 --test_start_seed 20000 --gen_seeds 0 --bon_free 8 --bon_signal cs_prefix \
    --log_diagnostics > $OUT/calib_h.log 2>&1 || { log "calib_h FAILED (see $OUT/calib_h.log)"; exit 1; }
  $PY scripts/rebuttal/aggregate_ablation.py calib $OUT/calib_h >> $OUT/calib_h.log 2>&1 \
    || { log "calib aggregation FAILED"; exit 1; }
fi
H_FIXED=$($PY -c "import json;print(json.load(open('$OUT/calib_h/h_calibration.json'))['h_median'])")
log "H_FIXED=$H_FIXED (frozen)"

# ---- 2. main variants, same 500 scenes and seed list; order = priority ----
run() {  # name, flags...
  local name=$1; shift
  if [ -f $OUT/main/$name/.done ]; then log "skip $name (done)"; return 0; fi
  log "start $name"
  local t0=$(date +%s)
  $PY scripts/eval_policy_sim.py -c $CKPT -o $OUT/main/$name --force $S3 "$@" > $OUT/main/$name.log 2>&1
  local rc=$?
  [ $rc -eq 0 ] && touch $OUT/main/$name/.done
  log "end $name rc=$rc wall=$(( $(date +%s) - t0 ))s"
}
jobs_wait() { while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do wait -n; done; }

jobs_wait; run base            --bon_free 1 &
jobs_wait; run cs_prefix       --bon_free 8 --bon_signal cs_prefix --log_diagnostics --cs_fixed_h $H_FIXED &
jobs_wait; run cs_endpoint     --bon_free 8 --bon_signal cs_endpoint &
jobs_wait; run medoid_prefix   --bon_free 8 --bon_signal medoid_prefix &
jobs_wait; run cs_prefix_fixed --bon_free 8 --bon_signal cs_prefix_fixed --cs_fixed_h $H_FIXED &
wait

# ---- 3. aggregation (strict) ----
D="$OUT/main/base $OUT/main/cs_prefix $OUT/main/cs_endpoint $OUT/main/medoid_prefix $OUT/main/cs_prefix_fixed"
for C in base cs_prefix; do
  $PY scripts/rebuttal/aggregate_ablation.py ablation $D --control $C --out $OUT/main/vs_${C}_auto.md \
    > $OUT/main/agg_vs_$C.txt 2>&1
  log "aggregate --control $C exit=$?  (see $OUT/main/agg_vs_$C.txt)"
done
log "DONE"
