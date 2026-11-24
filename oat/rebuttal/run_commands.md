# AAMAS-27 rebuttal — CS ablation on LIBERO-LONG: run commands

Branch `aamas27_rebuttal` = `52e55e6` (code that produced the S3 selector numbers) + ablation/logging
patch. All commands from `OAT_BLT_research/oat` (eval_policy_sim.py chdirs there anyway).

## 0. Setup (once)

```bash
cd /workspace/my_project/OAT_BLT_research
# branch is local-only until pushed. Either fetch it after the push, or carry it as a bundle:
#   (local)     git bundle create aamas27_rebuttal.bundle 52e55e6..aamas27_rebuttal   # copy file to container
#   (container) git fetch /path/aamas27_rebuttal.bundle aamas27_rebuttal:aamas27_rebuttal
git checkout aamas27_rebuttal
git log --oneline -1          # must match `git log --oneline -1` on the local branch
git lfs pull --include="oat/my_models/policy_ep-0250_sr-0.596.ckpt,oat/my_models/tokenizer_ep-0950_mse-0.002.ckpt"
cd oat
sha256sum my_models/policy_ep-0250_sr-0.596.ckpt
# expected 0924c1fbdb6ab973eb15fad70d8a986ef8dc596168a14f6db4acbd8d8bfa8405  (S3 checkpoint, NOT backup/)

# checkpoint loads? where does it look for the tokenizer? normalizer present?
uv run python -c "
import torch, dill
p = torch.load(open('my_models/policy_ep-0250_sr-0.596.ckpt','rb'), pickle_module=dill)
cfg = p['cfg']; print(cfg.policy.action_tokenizer); print(cfg.task.policy.env_runner)
print([k for k in p['state_dicts']['model'] if 'normalizer' in k][:10])"

uv run python tests/test_cs_ablation.py          # selector equivalence checks, no sim
```

Common env for every run below (GPU1 is busy with someone else's job):

```bash
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
CKPT=my_models/policy_ep-0250_sr-0.596.ckpt
OUT=eval_out/rebuttal_cs_ablation
# S3 protocol flags. n_test/n_parallel_envs/test_start_seed fix the (task, scene) map — do not change.
# --bon_free is passed per run (8 for selectors, 1 for the single-candidate base agent).
S3="--use_k_tokens 8 --entropy_threshold 0 --temperature 1.0 --topk 10 \
    --selector_seed 0 --n_test 500 --n_parallel_envs 4 --test_start_seed 10000"
```

## 1. Smoke (20 episodes/variant; NOT part of results)

`n_test 20` changes the task layout, so smoke scenes are not the main scenes — that is fine.
`base` is included so the N=1 path and `--control base` aggregation are exercised before 500 scenes.

```bash
SM="--use_k_tokens 8 --entropy_threshold 0 --temperature 1.0 --topk 10 --selector_seed 0 \
    --n_test 20 --n_parallel_envs 4 --test_start_seed 10000 --gen_seeds 0"
mkdir -p $OUT/smoke
/usr/bin/time -v uv run python scripts/eval_policy_sim.py -c $CKPT -o $OUT/smoke/base --force $SM \
  --bon_free 1 2>&1 | tee $OUT/smoke/base.log
for V in cs_prefix cs_endpoint; do
  /usr/bin/time -v uv run python scripts/eval_policy_sim.py -c $CKPT -o $OUT/smoke/$V --force $SM \
    --bon_free 8 --bon_signal $V --log_diagnostics 2>&1 | tee $OUT/smoke/$V.log
done
D="$OUT/smoke/base $OUT/smoke/cs_prefix $OUT/smoke/cs_endpoint"
uv run python scripts/rebuttal/aggregate_ablation.py ablation $D --control base      --n_boot 200; echo "exit=$?"
uv run python scripts/rebuttal/aggregate_ablation.py ablation $D --control cs_prefix --n_boot 200; echo "exit=$?"
```
Gate to step 3 (SR on these 20 episodes is NOT a criterion). Both aggregations must exit 0 in
strict mode. Strict mode fails on:
- status != ok (incl. `hidden_reset`), unfinished episodes, duplicates;
- invalid/mismatching scene hashes or differing (task, init_id, repeat_seed) key sets;
- generated tokens K != 8 or executed chunk length R != 16 in any episode (`k_tokens`/`r_exec`
  columns; checked for base too);
- any executed-index mismatch vs the variant's own shadow decision, or decision R != 16.

Also eyeball: 20 rows per variant, policy_calls ≈ 20–35, wall-clock per variant (throughput).
`base` has no selector_decisions.csv — expected.

Scene protocol (v2, differs from the historical S3 runs): the final reset of every episode is
seeded with its seed label, and the vector-env worker clears a pending autoreset on explicit
reset (upstream bug: with shared_memory=False, an env whose previous episode ended with a
pending autoreset re-reset on its first step, discarding the first action chunk and changing the
scene; whether that happened depended on how the previous batch ended, so its effect on the
historical numbers is not assumed to be equal across methods). Compare only new-protocol runs with
each other. Report as: same checkpoint and selection settings, corrected env reset, common scene
pool for all variants.

Determinism check (optional, cheap): rerun `cs_prefix` smoke into `$OUT/smoke/cs_prefix_rerun`
and diff the `success` / `policy_calls` columns.

## 2. Bandwidth calibration for cs_prefix_fixed (held-out scenes, before any eval)

Scene seeds 20000+ (disjoint from the 10000–10499 evaluation labels; workers are seeded from
different values, so the scene sequence is different). 40 episodes = 4 per task.

```bash
uv run python scripts/eval_policy_sim.py -c $CKPT -o $OUT/calib_h --force \
  --use_k_tokens 8 --entropy_threshold 0 --temperature 1.0 --topk 10 --bon_free 8 --selector_seed 0 \
  --n_test 40 --n_parallel_envs 4 --test_start_seed 20000 \
  --bon_signal cs_prefix --gen_seeds 0 --log_diagnostics 2>&1 | tee $OUT/calib_h.log
uv run python scripts/rebuttal/aggregate_ablation.py calib $OUT/calib_h   # -> h_calibration.json
```
Freeze `h_median` → `H_FIXED` and record it in the report before running step 3c.

## 3. Main runs (500 scenes; same generation seed list for every variant)

Order: block A (base, cs_prefix, cs_endpoint) first, then medoid_prefix, then cs_prefix_fixed.
All runs: same 500 scenes, same generation seed list.

```bash
SEEDS=0          # minimum; target "0,1,2" if throughput allows (decide BEFORE looking at results)
# A1) single-candidate base agent (N=1): does selection still beat base under the fixed protocol?
uv run python scripts/eval_policy_sim.py -c $CKPT -o $OUT/main/base --force $S3 \
  --bon_free 1 --gen_seeds $SEEDS 2>&1 | tee $OUT/main/base.log
# A2) control + same-candidate shadow log (CS vs medoid / endpoint / fixed)
#     (omit --cs_fixed_h if H_FIXED is not frozen yet; shadow fixed idx is then -1)
uv run python scripts/eval_policy_sim.py -c $CKPT -o $OUT/main/cs_prefix --force $S3 --bon_free 8 \
  --bon_signal cs_prefix --gen_seeds $SEEDS --log_diagnostics --cs_fixed_h $H_FIXED 2>&1 | tee $OUT/main/cs_prefix.log
# A3)
uv run python scripts/eval_policy_sim.py -c $CKPT -o $OUT/main/cs_endpoint --force $S3 --bon_free 8 \
  --bon_signal cs_endpoint --gen_seeds $SEEDS 2>&1 | tee $OUT/main/cs_endpoint.log
# B)
uv run python scripts/eval_policy_sim.py -c $CKPT -o $OUT/main/medoid_prefix --force $S3 --bon_free 8 \
  --bon_signal medoid_prefix --gen_seeds $SEEDS 2>&1 | tee $OUT/main/medoid_prefix.log
# C)
uv run python scripts/eval_policy_sim.py -c $CKPT -o $OUT/main/cs_prefix_fixed --force $S3 --bon_free 8 \
  --bon_signal cs_prefix_fixed --cs_fixed_h $H_FIXED --gen_seeds $SEEDS 2>&1 | tee $OUT/main/cs_prefix_fixed.log

DIRS="$OUT/main/base $OUT/main/cs_prefix $OUT/main/cs_endpoint"   # add medoid_prefix / cs_prefix_fixed when done
# selection vs base (every variant paired against N=1)
uv run python scripts/rebuttal/aggregate_ablation.py ablation $DIRS --control base \
  --out $OUT/main/vs_base_auto.md
# CS ablation (every variant paired against cs_prefix)
uv run python scripts/rebuttal/aggregate_ablation.py ablation $DIRS --control cs_prefix \
  --out $OUT/main/vs_cs_prefix_auto.md
```
If `--cs_fixed_h` is not known yet when starting (a), omit it: the shadow `cs_prefix_fixed_idx`
is then -1; everything else is unaffected.

Parallel runs on one GPU are fine for SR but contaminate `inference_ms`; timing from these runs is
indicative only. Measured latency (P2) = separate runs on an idle GPU.

## Outputs per run dir
- `episodes.csv` — variant, task, init_id (global episode index), seed_label (=10000+init_id),
  repeat_seed, success, episode_finished, env_steps, policy_calls, inference_ms (sum of batched
  call wall-clock while the episode was active), batch_envs, scene_hash, status
- `selector_decisions.csv` (with --log_diagnostics) — per (episode, replan): executed_idx,
  cs_prefix_idx, medoid_prefix_idx, cs_endpoint_idx, cs_prefix_fixed_idx, h_prefix, h_endpoint,
  d_cs_medoid, d_cs_endpoint, n_unique_prefix
- `manifest.json` — argv, git commit + diff sha, checkpoint sha256, scene list sha256, GPU, torch
- `eval_log.json` — legacy aggregate (same format as S3)

## P1. Table 2 source: single-deviation K/R data -> kr_continuations.csv

Source npz = `oracle_grid_mt.npz` (branch_value_k.py grid mode; 120 states, 10 tasks, M=5,
K 1 vs 8, R 8 vs 32). Verified against Table 2: mean K +0.043, R -0.028, corr 0.079.
Only per-branch success fractions are stored -> CSV holds success counts out of M (script refuses
if p*M is not an integer).

```bash
.venv/bin/python scripts/rebuttal/export_kr.py -i <path>/oracle_grid_mt.npz \
  -o rebuttal/kr_continuations.csv --summary rebuttal/kr_summary.md \
  --M 5 --k_coarse 1 --k_full 8 --r_short 8 --r_long 32
```
Summary: per contrast mean, within-task bootstrap CI, improved/unchanged/worse counts (ties split
into 0/0, M/M, intermediate), exact sign test; two Table 2 contrasts + two supplementary ones.

## Scene protocol v3 (supersedes v2; P0 v2 results in eval_out/rebuttal_cs_ablation are NOT used)

P0 v2 failed the strict gate: 432/1500 (task, init_id, seed) keys had different scene hashes across
variants, only in the three tasks with articulated fixtures (KITCHEN_SCENE4 drawer, SCENE6
microwave, SCENE8 stove). Cause (upstream LIBERO, bddl_base_domain.py): each hard reset re-runs
_setup_placement_initializer, which appends Open/Close / TurnOn/Off samplers to
object_property_initializers without clearing it; on the k-th reset of an env object k samplers
draw from np.random, so the fixture joint and later placements depend on the env's reset count,
which depends on how earlier episodes in that worker ended (early success -> extra autoreset).
Fix: LiberoEnv.reset clears the list before each reset (oat/env/libero/env.py).

Verify (2 min): `diag_reset_determinism.py --manifest <any manifest> --tasks KITCHEN_SCENE4,KITCHEN_SCENE6,KITCHEN_SCENE8`
-> all cases same=True. Then `SEEDS=0,1,2 MAXJ=3 nohup bash rebuttal/main.sh > main_v3_driver.log 2>&1 &`
(new OUT eval_out/rebuttal_cs_ablation_v3; step 0 = scene gate on 100 episodes x 2 seeds; Random-8
is included as variant `random`; aggregations --control base / cs_prefix / random).
