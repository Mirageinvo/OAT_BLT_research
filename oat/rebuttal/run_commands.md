# AAMAS-27 rebuttal — CS ablation on LIBERO-LONG: run commands

Branch `aamas27_rebuttal` = `52e55e6` (code that produced the S3 selector numbers) + ablation/logging
patch. All commands from `OAT_BLT_research/oat` (eval_policy_sim.py chdirs there anyway).

## 0. Setup (once)

```bash
cd /workspace/my_project/OAT_BLT_research
git fetch origin && git checkout aamas27_rebuttal
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
S3="--use_k_tokens 8 --entropy_threshold 0 --temperature 1.0 --topk 10 --bon_free 8 \
    --selector_seed 0 --n_test 500 --n_parallel_envs 4 --test_start_seed 10000"
```

## 1. Smoke (10–20 episodes/variant; NOT part of results)

`n_test 20` changes the task layout, so smoke scenes are not the main scenes — that is fine.

```bash
for V in cs_prefix cs_endpoint; do
  /usr/bin/time -v uv run python scripts/eval_policy_sim.py -c $CKPT -o $OUT/smoke/$V --force \
    --use_k_tokens 8 --entropy_threshold 0 --temperature 1.0 --topk 10 --bon_free 8 --selector_seed 0 \
    --n_test 20 --n_parallel_envs 4 --test_start_seed 10000 \
    --bon_signal $V --gen_seeds 0 --log_diagnostics 2>&1 | tee $OUT/smoke/$V.log
done
uv run python scripts/rebuttal/aggregate_ablation.py ablation $OUT/smoke/cs_prefix $OUT/smoke/cs_endpoint --n_boot 200
```
Check: episodes.csv has 20 rows, scene hashes equal across the two variants, policy_calls ≈ 20–35,
selector_decisions.csv non-empty, wall-clock per variant → throughput estimate.

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

```bash
SEEDS=0          # minimum; target "0,1,2" if throughput allows (decide BEFORE looking at results)
# a) control + same-candidate shadow log (CS vs medoid / endpoint / fixed)
uv run python scripts/eval_policy_sim.py -c $CKPT -o $OUT/main/cs_prefix --force $S3 \
  --bon_signal cs_prefix --gen_seeds $SEEDS --log_diagnostics --cs_fixed_h $H_FIXED 2>&1 | tee $OUT/main/cs_prefix.log
# b)
uv run python scripts/eval_policy_sim.py -c $CKPT -o $OUT/main/cs_endpoint --force $S3 \
  --bon_signal cs_endpoint --gen_seeds $SEEDS 2>&1 | tee $OUT/main/cs_endpoint.log
# c)
uv run python scripts/eval_policy_sim.py -c $CKPT -o $OUT/main/medoid_prefix --force $S3 \
  --bon_signal medoid_prefix --gen_seeds $SEEDS 2>&1 | tee $OUT/main/medoid_prefix.log
# d)
uv run python scripts/eval_policy_sim.py -c $CKPT -o $OUT/main/cs_prefix_fixed --force $S3 \
  --bon_signal cs_prefix_fixed --cs_fixed_h $H_FIXED --gen_seeds $SEEDS 2>&1 | tee $OUT/main/cs_prefix_fixed.log

uv run python scripts/rebuttal/aggregate_ablation.py ablation \
  $OUT/main/cs_prefix $OUT/main/cs_endpoint $OUT/main/medoid_prefix $OUT/main/cs_prefix_fixed \
  --out $OUT/main/rebuttal_results_auto.md
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
