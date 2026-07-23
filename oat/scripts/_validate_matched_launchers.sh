#!/usr/bin/env bash
# Dry-validate paper matched launch scripts (MetaWorld + RoboMimic).
# Does NOT run evals — only bash -n, path existence, ENV_TASK wiring.
set -euo pipefail
cd /workspace/oat

ok=0
bad=0
warn=0
pass() { echo "OK  $*"; ok=$((ok+1)); }
fail() { echo "FAIL $*"; bad=$((bad+1)); }
note() { echo "WARN $*"; warn=$((warn+1)); }

echo "=== syntax ==="
for f in \
  scripts/cluster_matched_triplet.sh \
  scripts/cluster_matched_paper_wave2_awr.sh \
  scripts/cluster_launch_matched_paper_wave.sh \
  scripts/cluster_matched_baseline.sh \
  scripts/_launch_lift_ep1400_matched.sh \
  scripts/_launch_lift_ep1400_awr_eval.sh \
  scripts/_launch_coffee_w2_parallel.sh \
  scripts/_launch_coffee_pull_matched_after_eval.sh \
  scripts/_launch_square_matched_parallel.sh \
  scripts/_launch_square_matched_on_plateau.sh \
  scripts/cluster_gpu_env.sh \
  scripts/eval_policy_sim.py \
  scripts/collect_awr_dataset.py \
  scripts/train_awr.py
do
  if [[ ! -f "$f" ]]; then fail "missing $f"; continue; fi
  if [[ "$f" == *.py ]]; then
    python -m py_compile "$f" && pass "py_compile $f" || fail "py_compile $f"
  else
    bash -n "$f" && pass "bash -n $f" || fail "bash -n $f"
  fi
done

echo
echo "=== paper BASE_CKPT defaults (triplet) ==="
resolve_triplet() {
  local SUITE="$1"
  local BASE_CKPT="" AWR_CKPT="" ENV_TASK="" SUITE_DIR=""
  case "${SUITE}" in
    lift)
      BASE_CKPT="${BASE_CKPT:-output/20260719/144024_train_oatpolicy_lift_N200/checkpoints/ep-0900_sr-0.930.ckpt}"
      ENV_TASK=""; SUITE_DIR="lift" ;;
    can)
      BASE_CKPT="${BASE_CKPT:-output/20260706/173343_train_oatpolicy_can_N200/checkpoints/ep-1700_sr-0.940.ckpt}"
      ENV_TASK=""; SUITE_DIR="can" ;;
    square)
      BASE_CKPT="${BASE_CKPT:-output/20260720/215024_train_oatpolicy_square_N200/checkpoints/ep-0700_sr-0.420.ckpt}"
      ENV_TASK=""; SUITE_DIR="square" ;;
    coffee-pull)
      BASE_CKPT="${BASE_CKPT:-output/20260720/090816_train_oatpolicy_mw-coffee-pull_st_N50/checkpoints/ep-1000_sr-0.432.ckpt}"
      ENV_TASK="coffee-pull"; SUITE_DIR="coffee-pull" ;;
    stick-pull)
      BASE_CKPT="${BASE_CKPT:-output/20260711/134439_train_oatpolicy_mw-stick-pull_st_N50/checkpoints/ep-0800_sr-0.212.ckpt}"
      ENV_TASK="stick-pull"; SUITE_DIR="stick-pull" ;;
    disassemble)
      BASE_CKPT="${BASE_CKPT:-output/20260711/134440_train_oatpolicy_mw-disassemble_st_N50/checkpoints/ep-1400_sr-0.700.ckpt}"
      ENV_TASK="disassemble"; SUITE_DIR="disassemble" ;;
    box-close)
      BASE_CKPT="${BASE_CKPT:-output/20260711/134439_train_oatpolicy_mw-box-close_st_N50/checkpoints/ep-2000_sr-0.552.ckpt}"
      ENV_TASK="box-close"; SUITE_DIR="box-close" ;;
    mt4)
      BASE_CKPT="${BASE_CKPT:-output/20260708/032431_train_oatpolicy_mw-mt4_N50/checkpoints/ep-0450_sr-0.280.ckpt}"
      ENV_TASK="mt4"; SUITE_DIR="mt4" ;;
  esac
  printf "%s|%s|%s|%s\n" "$SUITE" "$SUITE_DIR" "${ENV_TASK:--}" "$BASE_CKPT"
}

for s in lift can square coffee-pull stick-pull disassemble box-close; do
  IFS='|' read -r suite sdir etask ckpt < <(resolve_triplet "$s")
  [[ "$etask" == "-" ]] && etask=""
  if [[ -n "$ckpt" && -f "$ckpt" ]]; then
    pass "triplet $suite ENV_TASK='${etask}' -> $ckpt"
  else
    fail "triplet $suite missing '$ckpt'"
  fi
  # MW must pass env_task; RM must not
  case "$s" in
    coffee-pull|stick-pull|disassemble|box-close)
      [[ -n "$etask" ]] && pass "MW $s has ENV_TASK" || fail "MW $s missing ENV_TASK"
      ;;
    lift|can|square)
      [[ -z "$etask" ]] && pass "RM $s ENV_TASK empty (cfg-driven)" || fail "RM $s unexpected ENV_TASK=$etask"
      ;;
  esac
done

# paper lift B override
LB=output/20260719/144024_train_oatpolicy_lift_N200/checkpoints/ep-1400_sr-0.950.ckpt
[[ -f "$LB" ]] && pass "lift B paper BASE $LB" || fail "lift B missing $LB"
[[ -f my_models/awr_s10000_lift_ep1400.ckpt ]] && pass "lift B AWR ckpt" || note "lift B AWR ckpt not yet"

echo
echo "=== wave2 defaults + Wave1 prereqs ==="
for s in box-close disassemble coffee-pull stick-pull can lift square; do
  IFS='|' read -r suite sdir etask ckpt < <(resolve_triplet "$s")
  [[ "$etask" == "-" ]] && etask=""
  w1="output/eval/matched_s10000/${sdir}"
  b="$w1/baseline_n5/eval_log.json"
  n="$w1/bon_n8_n5/eval_log.json"
  if [[ -n "$ckpt" && -f "$ckpt" ]]; then pass "wave2 base $s"; else fail "wave2 base $s missing '$ckpt'"; fi
  if [[ -f "$b" && -f "$n" ]]; then
    pass "wave1 ready $s"
  else
    note "wave1 incomplete $s (baseline=$([[ -f $b ]]&&echo Y||echo N) bon=$([[ -f $n ]]&&echo Y||echo N)) — wave2 would block"
  fi
done
# lift_ep1400 separate root
if [[ -f output/eval/matched_s10000/lift_ep1400/baseline_n5/eval_log.json \
   && -f output/eval/matched_s10000/lift_ep1400/bon_n8_n5/eval_log.json ]]; then
  pass "wave1 ready lift_ep1400"
else
  fail "lift_ep1400 wave1 incomplete"
fi

echo
echo "=== data (zarr) ==="
for p in \
  data/robomimic/lift_N200.zarr \
  data/robomimic/can_N200.zarr \
  data/robomimic/square_N200.zarr \
  data/metaworld/coffee-pull_N50.zarr \
  data/metaworld/stick-pull_N50.zarr \
  data/metaworld/disassemble_N50.zarr \
  data/metaworld/box-close_N50.zarr
do
  [[ -d "$p" ]] && pass "zarr $p" || fail "zarr $p"
done

echo
echo "=== known stale / traps ==="
note "SUITE=mt4 now hard-fails in triplet/baseline (deleted; not in paper)"
note "cluster_matched_baseline.sh = Table B lab (seed 1000, n=3); paper = cluster_matched_triplet.sh (seed 10000, n=5)"
note "triplet SUITE=lift default = ep-0900 (run A). Paper TopK = ep-1400 via _launch_lift_ep1400_*.sh"

echo
echo "=== SUMMARY ok=$ok warn=$warn fail=$bad ==="
[[ "$bad" -eq 0 ]]
