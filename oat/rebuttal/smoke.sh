#!/usr/bin/env bash
# Smoke for the AAMAS-27 rebuttal CS ablation: base / cs_prefix / cs_endpoint, 20 episodes each.
# Run from OAT_BLT_research/oat:   nohup bash rebuttal/smoke.sh > smoke_driver.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-0}
PY=${PY:-.venv/bin/python}
CKPT=my_models/policy_ep-0250_sr-0.596.ckpt
OUT=eval_out/rebuttal_cs_ablation/smoke
SM="--use_k_tokens 8 --entropy_threshold 0 --temperature 1.0 --topk 10 --selector_seed 0 \
    --n_test 20 --n_parallel_envs 4 --test_start_seed 10000 --gen_seeds 0"
mkdir -p $OUT

run() {  # name, extra flags...
  local name=$1; shift
  echo "[$(date +%T)] start $name"
  local t0=$(date +%s)
  $PY scripts/eval_policy_sim.py -c $CKPT -o $OUT/$name --force $SM "$@" > $OUT/$name.log 2>&1
  local rc=$?
  echo "[$(date +%T)] end $name rc=$rc wall=$(( $(date +%s) - t0 ))s"
}

run base        --bon_free 1
run cs_prefix   --bon_free 8 --bon_signal cs_prefix   --log_diagnostics
run cs_endpoint --bon_free 8 --bon_signal cs_endpoint --log_diagnostics

D="$OUT/base $OUT/cs_prefix $OUT/cs_endpoint"
for C in base cs_prefix; do
  $PY scripts/rebuttal/aggregate_ablation.py ablation $D --control $C --n_boot 200 \
    > $OUT/agg_vs_$C.txt 2>&1
  echo "[$(date +%T)] aggregate --control $C exit=$?  (see $OUT/agg_vs_$C.txt)"
done
echo "[$(date +%T)] DONE"
