#!/usr/bin/env bash
# AAMAS-27 rebuttal: Random-8 control vs CS-8 (cs_prefix from the P0 main block).
# Run AFTER the P0 main block has finished, from OAT_BLT_research/oat:
#   R8_SEEDS=0,1,2 nohup bash rebuttal/random8.sh > random8_driver.log 2>&1 &
# The checkout may be newer than P0 only if the commits in between touch nothing but
# oat/rebuttal/ and oat/scripts/rebuttal/ (checked below; eval/runner/env code must be identical).
# Steps: smoke (20 ep, strict aggregation vs P0 smoke cs_prefix) -> main (500 scenes) ->
# paired view (manifest/hash checks) -> strict aggregation --control random (and --control base).
set -u
cd "${OAT_DIR:-/workspace/my_project/OAT_BLT_research/oat}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-0}
PY=${PY:-.venv/bin/python}
CKPT=my_models/policy_ep-0250_sr-0.596.ckpt
P0=eval_out/rebuttal_cs_ablation
ROOT=eval_out/rebuttal_random8
SEEDS=${R8_SEEDS:-0,1,2}               # fixed before looking at results
TAG=seed_${SEEDS//,/-}
SMOKE=$ROOT/smoke/random_8
RUN=$ROOT/main/$TAG/random_8
VIEW=$ROOT/paired_$TAG
INCLUDE_BASE=${R8_INCLUDE_BASE:-1}
COMMON="--use_k_tokens 8 --entropy_threshold 0 --temperature 1.0 --topk 10 --n_parallel_envs 4
        --test_start_seed 10000 --selector_seed 0 --bon_free 8 --bon_signal random
        --bon_prefix_k 0 --bon_first_temp 0"
log() { echo "[$(date '+%F %T')] $*"; }
die() { log "FAILED: $*"; exit 1; }

log "SEEDS=$SEEDS commit=$(git rev-parse --short HEAD) dirty=[$(git status --short --untracked-files=no | tr '\n' ' ')]"
test -x "$PY" || die "no $PY"
P0_COMMIT=$($PY -c "import json;print(json.load(open('$P0/main/cs_prefix/manifest.json'))['git_commit'])")
# a newer commit is fine only if nothing outside the rebuttal helper dirs changed since P0
CHANGED=$(git diff --name-only $P0_COMMIT HEAD)
BAD=$(echo "$CHANGED" | grep -vE '^(oat/rebuttal/|oat/scripts/rebuttal/)' | grep -v '^$')
log "P0 commit=${P0_COMMIT:0:7}; changed since P0: $(echo $CHANGED)"
[ -z "$BAD" ] || die "code changed since P0 outside rebuttal dirs: $BAD"
mkdir -p $ROOT/smoke $ROOT/main/$TAG

# ---- 1. smoke: 20 episodes, paired with the P0 smoke cs_prefix (same smoke protocol) ----
if [ ! -f $SMOKE/.ok ]; then
  test -f $P0/smoke/cs_prefix/episodes.csv || die "no P0 smoke cs_prefix"
  rm -rf $SMOKE
  log "start smoke"
  $PY scripts/eval_policy_sim.py -c $CKPT -o $SMOKE $COMMON --n_test 20 --gen_seeds 0 \
    > $ROOT/smoke/random_8.log 2>&1 || die "smoke run (see $ROOT/smoke/random_8.log)"
  $PY scripts/rebuttal/aggregate_ablation.py ablation $P0/smoke/cs_prefix $SMOKE --control random \
    --expect_k 8 --expect_r 16 --expect_seeds 0 --expect_n_test 20 --n_boot 200 \
    > $ROOT/smoke/integrity.txt 2>&1 || die "smoke aggregation (see $ROOT/smoke/integrity.txt)"
  touch $SMOKE/.ok
  log "smoke OK"
else
  log "skip smoke (ok)"
fi

# ---- 2. main: 500 scenes, generation seeds $SEEDS; never appended to a partial run ----
if [ ! -f $RUN/.done ]; then
  [ -e $RUN ] && { mv $RUN $RUN.partial_$(date +%s); log "moved partial run aside"; }
  log "start main"
  t0=$(date +%s)
  $PY scripts/eval_policy_sim.py -c $CKPT -o $RUN $COMMON --n_test 500 --gen_seeds $SEEDS \
    > $ROOT/main/$TAG/random_8.log 2>&1 || die "main run (see $ROOT/main/$TAG/random_8.log)"
  git rev-parse HEAD > $RUN/.done
  log "end main wall=$(( $(date +%s) - t0 ))s"
fi

# ---- 3. paired view + strict aggregation (needs finished P0 cs_prefix [and base]) ----
for v in cs_prefix $( [ "$INCLUDE_BASE" = 1 ] && echo base ); do
  test -f $P0/main/$v/.done || die "P0 $v not finished"
done
IN="$P0/main/cs_prefix $RUN"; [ "$INCLUDE_BASE" = 1 ] && IN="$IN $P0/main/base"
rm -rf $VIEW
$PY scripts/rebuttal/paired_view.py $VIEW $SEEDS $IN > $ROOT/paired_view_$TAG.txt 2>&1 \
  || die "paired view checks (see $ROOT/paired_view_$TAG.txt)"
DIRS="$VIEW/cs_prefix $VIEW/random"; [ "$INCLUDE_BASE" = 1 ] && DIRS="$DIRS $VIEW/base"
AGG="--expect_k 8 --expect_r 16 --expect_seeds $SEEDS --expect_n_test 500 --n_boot 10000"
FAIL=0
for C in random $( [ "$INCLUDE_BASE" = 1 ] && echo base ); do
  $PY scripts/rebuttal/aggregate_ablation.py ablation $DIRS $AGG --control $C \
    --out $VIEW/vs_$C.md > $VIEW/agg_vs_$C.txt 2>&1
  rc=$?; log "aggregate --control $C exit=$rc (see $VIEW/agg_vs_$C.txt)"
  [ $rc -ne 0 ] && FAIL=1
done
[ $FAIL -eq 0 ] || die "aggregation"
log "ALL OK"
