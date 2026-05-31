# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a research repository containing two independent deep learning projects:

- **`blt/`** — Byte Latent Transformer (Meta Research): a byte-level LLM that dynamically groups bytes into variable-length patches using entropy, matching tokenization-based LLMs at scale.
- **`oat/`** — Ordered Action Tokenization (Harvard/Stanford): discrete action tokenization for robotic policy learning on the LIBERO benchmark.

---

## BLT (Byte Latent Transformer)

### Setup

```bash
cd blt
uv sync                                         # recommended
# or: bash setup/create_env.sh                 # conda alternative
python download_blt_weights.py --model blt-1b  # or blt-7b
```

### Common Commands

```bash
# Quick demo (text generation)
python demo.py

# Training
python -m bytelatent.train --config bytelatent/configs/<config>.yaml

# Evaluation
python -m bytelatent.eval

# Lint (black + isort)
bash dev/lint.sh

# Run all tests
cd blt && python -m pytest bytelatent/

# Run a single test file
python -m pytest bytelatent/test_blt.py

# Run a single test by name
python -m pytest bytelatent/test_blt.py::test_function_name
```

### Architecture

The core innovation is **entropy-based dynamic patching**: bytes are grouped into variable-length patches based on a learned entropy model, so high-entropy (unpredictable) regions get more compute.

**Forward pass data flow** (defined in `model/blt.py`):
1. Raw bytes → `LocalEncoder` (`model/local_models.py`): lightweight transformer over bytes, produces per-byte hidden states.
2. Patch boundaries determined by `data/patcher.py` (entropy/BPE/space/static/byte modes) using the small `entropy_model.py` network.
3. Byte hiddens aggregated → patch embeddings → `LatentTransformer` (`model/latent_transformer.py`): the large global model doing most of the compute.
4. Patch hiddens → `LocalDecoder` (`model/local_models.py`): cross-attends from patch representations back to byte positions to predict next bytes.

**Vocabulary**: 256 raw byte values + special tokens (PAD, BOS, EOS, BOE) starting at `OFFSET=3` — giving `vocab_size=260`. Defined in `tokenizers/constants.py`.

**Data pipeline**: iterators in `data/iterators/` form a chain — `ArrowIterator` → `PreprocessIterator` → `PackingIterator` → `SamplingIterator` → `MultiprocessIterator`. Each is stateful for resumable training.

**Training** (`train.py`): FSDP with optional tensor parallelism (`tp_size`). SLURM-aware via `submitit`; use `bytelatent/stool.py` to submit jobs. Float8 quantization supported via `float8.py`.

**Config system**: YAML under `bytelatent/configs/` parsed with omegaconf, CLI flags override YAML. All hyperparameters live in `TrainArgs`/`DistributedArgs` dataclasses in `args.py`.

---

## OAT (Ordered Action Tokenization)

### Setup

```bash
cd oat
uv sync
git submodule update --init --recursive        # fetch third_party/LIBERO
python scripts/convert_libero_dataset.py       # HDF5 → Zarr conversion
```

### Common Commands

```bash
# Train OAT tokenizer (add `accelerate launch` for multi-GPU)
uv run accelerate launch scripts/run_workspace.py --config-name=train_oattok task/tokenizer=libero/libero10

# Train OAT policy
uv run accelerate launch scripts/run_workspace.py --config-name=train_oatpolicy task/policy=libero/libero10

# Evaluate policy in simulation
uv run scripts/eval_policy_sim.py --checkpoint path/to/ckpt --output_dir path/to/out

# Compose multi-task dataset
python scripts/compose_libero_multitask_dataset.py
```

There are no unit tests; correctness is validated via LIBERO simulation rollouts in `env_runner/libero_runner.py`.

### Architecture

OAT has a two-stage training pipeline: **tokenizer training** then **policy training**.

**Tokenizer stage** (`workspace/train_oattok.py`):
- VQ-VAE: `RegisterEncoder` → FSQ quantizer (`tokenizer/oat/quantizer/fsq.py`) → `SinglePassDecoder`. FSQ default levels `[8,5,5,5]` give a codebook of 1000 entries.
- Trained on reconstruction loss; checkpoint saved for frozen use in stage 2.
- Alternatives with matching config/workspace/policy triplets: FastTok, BinTok, QuestTok (each in `tokenizer/<name>/`, `workspace/train_<name>tok.py`, `config/train_<name>tok.yaml`).

**Policy stage** (`workspace/train_policy.py`):
- `FusedObservationEncoder` (`perception/fused_obs_encoder.py`) merges `RobomimicRgbEncoder` (CNN vision) + `ProjectionStateEncoder` (MLP state) into a shared embedding.
- Fused embedding → transformer backbone → predicts discrete action token indices from the frozen tokenizer's codebook.
- Two policy heads available: **autoregressive** (`model/autoregressive/transformer_cache.py`, with KV-cache for inference; non-cache variant in `transformer.py`) and **diffusion** (`model/diffusion/`, uses `ConditionalUNet1D` or `TransformerForDiffusion`).
- Evaluation checkpointing keeps top-k by LIBERO task success rate.

**Config system**: Hydra-based under `oat/config/`, with task overrides in `config/task/tokenizer/` and `config/task/policy/`. The `scripts/run_workspace.py` entry point instantiates the workspace class via Hydra. Distributed training uses HuggingFace Accelerate. SLURM scripts are in `slurm/`.

---

## Active research: adaptive token budget for OAT

Goal: cut inference cost of `OATPolicy` by predicting *how many* action tokens are actually needed per observation, instead of always generating the full `max_seq_len` (=8, i.e. `num_registers` of the tokenizer that `policy_ep-0250` was trained with).

### Branches

- `first_improvement` — introduced `OATPolicy.predict_action_adaptive` in `oat/policy/oatpolicy.py`: greedy autoregressive generation with **per-step entropy threshold** early-stopping. `libero_runner.py` was switched from `predict_action` to `predict_action_adaptive`; runner now also accumulates `n_tokens` and reports `mean_tokens_used` in the eval log. `scripts/eval_policy_sim.py` prints it alongside `mean_success_rate`.
- `tok_num_generator` (current branch) — replace the entropy threshold with a **learned complexity predictor** that estimates min-k per observation. Labels are MSE-based, not simulation-based. Dataset collection script is done; predictor model is next.

### Label generation (MSE-based)

Implemented in `oat/scripts/collect_min_k_dataset.py`:
- Iterate the policy's training `ZarrDataset` (same Hydra config used to train the checkpoint).
- Per batch: encode obs → features; greedy-decode all `max_seq_len` tokens; for each `k = 1..max_seq_len`, detokenize the first `k` tokens and compute **normalized RMS per-dim error** in the action-normalizer space `[-1, 1]`: `err(k) = sqrt(mean((norm(a_pred(k)) - norm(a_gt))^2))`. This makes ε interpretable as average fraction of each dim's data range.
- GT actions are sliced to `n_action_steps` (=16) before comparison — the dataset stores 32 steps but policy only predicts 16.
- Save `features [N, To, d]`, `errors [N, max_k]`, `task_uids` to `.npz`. Storing the full `errors` array (not just `min_k`) so ε can be varied later without re-running.
- Default ε for stats: **10%**. `min_k = first k where err < ε`, else `max_k`. Script prints detailed stats: per-k error percentiles, cumulative success fractions, effective-k histogram.
- Features (post-`obs_encoder`) are saved rather than raw obs: dataset is policy-specific anyway, and obs are large (128×128 images).

Run:
```bash
cd oat && uv run python scripts/collect_min_k_dataset.py \
    -c my_models/policy_ep-0250_sr-0.596.ckpt \
    -o my_datasets/libero10_min_k_features.npz
```

Output: `my_datasets/libero10_min_k_features.npz` with keys `features`, `errors`, optionally `task_uids`. Collected dataset (for `policy_ep-0250`): `features (124087, 2, 138)` = (N, To, obs_feature_dim), `errors (124087, 8)` = (N, max_k), `task_uids (124087, 2, 1)`.

### Methodological notes

- "Circularity" concern (using the trained policy to label data for its own helper) is intrinsic to the task — we are predicting a property *of this policy*. This is the same pattern used by CALM, Adaptive Computation Time, early-exit networks, and speculative decoding.
- MSE labels measure **fidelity of the k-token reconstruction to the ground-truth demonstration** (`err(k) = RMS(norm(a_pred(k)) - norm(a_gt))`), i.e. min_k = fewest tokens to land within ε of the demo. Caveat: this is *not* the same as "policy output stabilized" (which would compare `a_pred(k)` to `a_pred(max_k)`); a sample where the policy is simply wrong even at `max_k` never reaches ε and is labeled fail, so the "fail rate" mixes "needs many tokens" with "policy missed". Both also differ from "policy succeeds at the task" (the simulation-success label). The plan is to validate the learned predictor end-to-end via LIBERO success rate, treating MSE labels as the cheap-to-collect training signal.
- If MSE-trained predictor degrades success rate, fall back to simulation-success labels (≈8× more expensive: run eval with `use_k_tokens=1..8`, take per-episode min-k where success=True).

### Token-count predictor (model + training)

Status: trained on cluster (ε=0.10). Wired into `OATPolicy` for end-to-end eval (see below); **LIBERO success-rate validation still pending**. Chosen operating point: **`w=2.0`** (`my_models/token_count_predictor_w2.0.ckpt`).

- Model: `oat/oat/model/token_count_predictor.py` — `TokenCountPredictor`, a small MLP `features [B, To, d] → flatten(To·d) → hidden(256,256) → max_k logits` (classes = k=1..max_k). Feature z-score stats (`feat_mean`/`feat_std`) are stored as buffers so a loaded predictor normalizes raw `features` itself (self-contained inference). Helpers: `predict_k()` → k∈[1,max_k], `from_checkpoint()`.
- Training: `oat/scripts/train_token_count_predictor.py`. Loads the `.npz`, builds labels `min_k(ε) = first k with err<ε else max_k` (fail → full budget), standardizes on the train split, trains with CE.
  - **Asymmetry-aware:** under-predicting k (too few tokens) hurts action/success; over-predicting only costs latency. So checkpoint selection uses `safe_rate` (`pred≥true`), tie-broken by lower `mean_pred_k` — **not** raw accuracy. Reports `under_rate`, `safe_rate`, `mean_pred_k` vs `mean_true_k` vs `max_k`, pred/true histograms.
  - `--underpredict_weight >1` up-weights CE on currently under-predicted samples (safety bias); default 1.0 (plain CE baseline).
  - Run:
    ```bash
    cd oat && uv run python scripts/train_token_count_predictor.py \
        -i my_datasets/libero10_min_k_features.npz \
        -o my_models/token_count_predictor.ckpt --epsilon 0.10
    ```
Results — `--underpredict_weight` sweep (ε=0.10, val; `full_budget_within_eps=0.624` is the hard ceiling, since 31% of samples never reach ε at any k):

| `w` | `mean_pred_k` | AR steps saved | `frac_within_eps` | `success_retained` | `under_rate` | `over_rate` |
|-----|---------------|----------------|-------------------|--------------------|--------------|-------------|
| 1.0 | 4.00 | ~50% | 0.520 | 83.4% | 0.224 | 0.206 |
| 2.0 | 4.76 | ~40% | 0.547 | 87.7% | 0.154 | 0.314 |
| 4.0 | 6.02 | ~25% | 0.582 | 93.3% | 0.082 | 0.474 |

Findings:
- The predictor effectively **collapses to binary**: `pred_hist` is concentrated on k∈{1,2,8}, almost never k=3..7 (middle classes are ~14% of data and noisy → unpredictable from features). So it is an "easy (1/2) vs hard (8)" detector; `w` just shifts the decision boundary toward the safe side.
- `mean_err_realized` is nearly constant across the sweep (~0.135/0.133/0.129, p90 ~0.282): aggressive settings do **not** blow up action error; the extra `within_eps` misses are marginal samples near the ε boundary, not catastrophes.
- All offline numbers use **greedy** decoding + GT-demonstration labels, while the policy **samples** at inference (temp=1, topk=10) — treat as optimistic proxy; only LIBERO success (TODO #5) is decisive.
- Possible v2 if CE under-predicts too much: per-k binary "is k sufficient" heads → pick smallest sufficient k at inference, with a tunable decision threshold (no retrain to change it).

### Adaptive generation: entropy vs learned predictor

`OATPolicy.predict_action_adaptive` now **dispatches**: if a token-count predictor is attached (`policy.set_token_predictor(...)`), it calls `predict_action_predictor`; otherwise it runs the original entropy-threshold path. So the runner / eval call site is unchanged — attaching a predictor switches modes.

`predict_action_adaptive` (entropy mode):
- Greedy-ish AR generation **without KV-cache** (calls `self.model()` each step, not `self.model.generate()`). Conscious trade-off for simplicity (~3.5 ms/token overhead vs KV-cache version).
- Entropy computed on **full** distribution (before top-k masking), averaged across batch. Early stop: `if step >= 1 and entropy < threshold` (minimum 2 tokens; batch stops together).
- Default `entropy_threshold=2.75`. Entropy range: `[0, log(1000) ≈ 6.91]` nats (codebook size 1000).
- Returns: `{'action', 'action_pred', 'n_tokens', 'entropies'}`.

`predict_action_predictor` (learned mode):
- `features = obs_encoder(obs)` → `k_pred = token_predictor.predict_k(features)` (per-sample, clamped to `[1, use_k_tokens or max_seq_len]`).
- Generates `K = max(k_pred)` tokens for the whole batch via the **KV-cache** `self.model.generate()` (no per-step entropy needed → faster than entropy mode), then decodes each sample at its **own** `k_pred` via `OATTok.detokenize(tokens, eval_keep_k=k_pred)`. NB: batched AR latency is bound by `max(k_pred)`; with batch_size=1 it is exact per-sample.
- `detokenize` gained an optional `eval_keep_k: List[int]` (per-sample budget) for this.
- Returns: `{'action', 'action_pred', 'n_tokens' (batch-mean budget), 'k_pred'}`.
- The predictor is attached post-hoc (not a submodule), so it is excluded from the policy checkpoint and stays frozen.

### Eval pipeline

- `oat/scripts/eval_policy_sim.py` — CLI entry point. Flags: `-c`, `-o`, `-n`, `-d`, `--temperature`, `--topk`, `--use_k_tokens`, `--token_predictor` (path to a `TokenCountPredictor` ckpt → attaches it for learned adaptive generation). **No `--entropy_threshold` flag** — entropy mode always uses default 2.75. Adding that flag is a known TODO.
- `oat/oat/env_runner/libero_runner.py` — rollout runner. Calls `policy.predict_action_adaptive()`, accumulates `n_tokens` per step, reports `mean_tokens_used` in log.
- `oat/oat/workspace/train_policy.py:331` — validation also uses `predict_action_adaptive`.

### Latency benchmarking

`oat/my_scripts/measure_latency_adaptive.py` — measures inference latency per token-cap value.
- `--obs_from_checkpoint_dataset` flag loads real obs from val dataset (important: dummy random obs produce near-uniform entropy ~6.9, never triggering early stop).
- `--n_batches N` cycles through N distinct obs batches during timing.
- Prints per-step entropy percentiles and speedup table.

```bash
cd oat && uv run python my_scripts/measure_latency_adaptive.py \
    -c my_models/policy_ep-0250_sr-0.596.ckpt \
    --obs_from_checkpoint_dataset --n_batches 10
```

### Key files touched

- `oat/oat/policy/oatpolicy.py` — `predict_action` (KV-cache), `predict_action_adaptive` (entropy mode, dispatches to predictor when one is attached), `predict_action_predictor` (learned mode), `set_token_predictor`
- `oat/oat/tokenizer/oat/tokenizer.py` — `OATTok.detokenize` gained optional per-sample `eval_keep_k`
- `oat/oat/env_runner/libero_runner.py` — uses `predict_action_adaptive`, tracks `mean_tokens_used`
- `oat/scripts/eval_policy_sim.py` — prints `mean_tokens_used`
- `oat/scripts/collect_min_k_dataset.py` — label generation for token-count predictor
- `oat/oat/model/token_count_predictor.py` — `TokenCountPredictor` MLP (features → min_k)
- `oat/scripts/train_token_count_predictor.py` — trains the predictor from the `.npz`
- `oat/my_scripts/measure_latency_adaptive.py` — latency benchmarking with real obs

### End-to-end results (LIBERO, `policy_ep-0250`, 5 exps each)

| mode | SR | mean tokens | notes |
|------|-----|-------------|-------|
| full budget (k=8) | ~0.58 (paper) / 0.596 (train-eval) | 8 | reference |
| entropy threshold 2.75 | 0.501 ± 0.016 | 5.68 | heuristic baseline |
| learned predictor w=2.0 | 0.497 ± 0.017 | 5.31 | `token_count_predictor_w2.0.ckpt` |

Conclusions:
- **Learned predictor ≈ entropy heuristic in SR** (0.497 vs 0.501, CIs fully overlap) but uses **fewer tokens** (5.31 vs 5.68) → marginally Pareto-better, not a decisive win.
- **The ~8–10pp SR drop vs full budget is method-independent** (both adaptive schemes land at ~0.50). This points to *compounding error over the episode* — slightly worse per-chunk reconstruction accumulates over ~550 steps — which the single-chunk MSE label cannot capture. So polishing the predictor on MSE labels is unlikely to break past this ceiling.
- Decision pending: ~30% token savings for ~9pp (~15% rel.) success is expensive if success is the priority.

Speed note: eval is dominated by simulation (obs cameras rendered every step for all parallel envs). Use `MUJOCO_GL=egl` (GPU offscreen; training slurm sets it, eval did not), lower `n_test`/`n_test_vis` for iteration. (`eval_policy_sim.py` does not yet expose runner overrides via CLI.)

### TODO next

1. ~~**Analyze `collect_min_k_dataset.py` output**~~ — done; min_k distribution at ε=0.10 checked, looks good.
2. ~~**Build token-count predictor**~~ — model + training script written (`oat/oat/model/token_count_predictor.py`, `oat/scripts/train_token_count_predictor.py`). **Next: train on cluster and check `safe_rate`/`under_rate`/`mean_pred_k`.**
3. ~~**Wire predictor into `predict_action_adaptive`**~~ — done: `predict_action_predictor` + `set_token_predictor` + `eval_policy_sim.py --token_predictor` + `detokenize(eval_keep_k=...)`. Compiles; **not yet run in sim.**
4. **Add `--entropy_threshold` flag to `eval_policy_sim.py`** — for threshold sweep experiments.
5. ~~**End-to-end validation**~~ — done for full / entropy-2.75 / predictor-w2.0 (see "End-to-end results" above). Result: learned ≈ entropy in SR, both ~8–10pp below full budget; cost looks method-independent (compounding error).
6. **Map the knee: run predictor `w=4.0`** (~6 tokens, expect SR ~0.52–0.54) to complete the SR-vs-tokens frontier and pick the operating point.
7. **If a real tradeoff win is needed:** switch the predictor's training signal from MSE-to-demo labels to **simulation-success** labels (per-episode min-k where success holds) — directly targets task success and removes the compounding-error gap. ~8× more eval cost.
8. **Latency reality check:** `measure_latency_adaptive.py` at `batch_size=1` for true per-sample ms (batched runner latency is bound by `max(k_pred)`, not the mean).
