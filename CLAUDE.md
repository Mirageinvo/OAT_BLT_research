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
- **Idea — online binary stop/continue controller (sequential v2, generation-aware):** instead of predicting k from obs upfront, a per-step binary head decides "stop or generate one more token" conditioned on the **already-generated tokens** (reuse the AR model's last-layer hidden state — ~free — optionally + decoded prefix action `a_k` / `Δ(a_k,a_{k-1})`). This is the learned version of the entropy/info-gain stop and matches SkiP's refine-or-stop. Advantages: fixes the obs-only predictor's *generation-blindness*; per-step binary is a much easier target than 8-way `min_k` (sidesteps the {1,2,8} collapse; 8 labels/sample, no brittle argmin); tunable threshold without retrain. **Caveat:** it improves the *mechanism/trainability*, not the ceiling — if trained on MSE/self-consistency labels it's just a learned action-convergence stop (risks matching the heuristic that ≈ fixed-k in the gate). Payoff still rides on (a) per-sample headroom (gate) and (b) a **task-grounded** training signal. → pair this mechanism with task-grounded labels, not MSE.

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
- `oat/oat/model/token_count_predictor.py` — `TokenCountPredictor` MLP (features → min_k); `class_values` for 4-class {1,2,4,8}
- `oat/scripts/train_token_count_predictor.py` — trains the predictor; `--valid_budgets`, realized-error + safe-rate metrics
- `oat/scripts/per_timestep_recon_error.py` — Step 0: intra-chunk gain/excess concentration + gripper/jerk semantics (H-OAT premise check)
- `oat/oat/policy/oatpolicy.py` — also `predict_action_agnostic` + `set_agnostic_mix` (gate baseline); `predict_action_variable_r` (GATE 1 variable-R: convergence/random/fixed R-signal, dispatched by `adaptive_r` kwarg; R horizon resolved from decoded `action_pred.shape[1]`=32, NOT latent_horizon=8 — see horizon bug fix)
- `oat/scripts/diag_convergence_div.py` — offline (no-sim) convergence R-signal diagnostic: per-timestep `d_t` percentiles + threshold→R-distribution sweep (decides R-heterogeneity before sim)
- `oat/my_scripts/test_state_roundtrip.py` — verifies LIBERO/MuJoCo snapshot+restore (get_sim_state/set_state/regenerate_obs_from_state) is deterministic → prereq for value(k) branching (PASSED)
- `oat/scripts/branch_value_k.py` — Value-Guided OAT Phase-1 pilot: counterfactual branching (restore→exec k-chunk→continue k=8→success) to measure per-state value-gap `p(k=8)−p(k=1)` vs phase (contact) and vs reconstruction-gap → tests headline C1 `reconstruction(k)≠value(k)`
- `oat/oat/gymnasium_util/multistep_wrapper.py` — `step` breaks on all-NaN action row (variable-R sentinel)
- `oat/oat/env_runner/libero_runner.py` — logs in-sim `k_pred` histogram; variable-R: NaN-pad per-env R, `mean_r_exec` + R-hist, `episode_done` desync fix
- `oat/scripts/eval_policy_sim.py` — flags `--token_predictor`, `--entropy_threshold`, `--agnostic_mix`, `--n_action_steps` (fixed-R sweep), `--adaptive_r`/`--r_coarse_k`/`--r_min`/`--r_max`/`--r_threshold` (variable-R)
- `oat/my_scripts/measure_latency_adaptive.py` — latency benchmarking with real obs

### End-to-end results (LIBERO, `policy_ep-0250`, 5 exps each)

| mode | SR | mean tokens | notes |
|------|-----|-------------|-------|
| full budget (k=8) | ~0.58 (paper) / 0.596 (train-eval) | 8 | reference |
| learned predictor w=4.0 | 0.559 ± 0.014 (3 exp) | 6.29 | ~96% of full, −21% tokens |
| entropy threshold 2.75 | 0.501 ± 0.016 | 5.68 | heuristic baseline |
| learned predictor w=2.0 | 0.497 ± 0.017 | 5.31 | `token_count_predictor_w2.0.ckpt` |
| **fixed k=4** | **0.496 ± 0.025 (5 exp)** | 4.0 | valid budget — **dominates w=2.0 & entropy** |
| fixed k=5 | 0.496 ± 0.009 (3 exp) | 5.0 | INVALID baseline — untrained budget |
| fixed k=6 | 0.446 (2 exp) | 6.0 | INVALID baseline — untrained budget |

**GATE read with valid frontier {4,8} — mostly NEGATIVE.** In-pipeline anchors: fixed k=4 = 0.496 @ 4.0, fixed k=8 = 0.58 @ 8.0 (slope 0.021 SR/token). fixed k=4 **Pareto-dominates** predictor w=2.0 (0.497 @ 5.31) and entropy (0.501 @ 5.68): same SR, fewer tokens → at ~0.50 SR a constant k=4 beats both adaptive methods. High end: predictor w=4.0 (0.559 @ 6.29) is **+0.015 above** the {4,8} obs-agnostic mixing line (0.544 at 6.29) → ~1.4σ, suggestive but not significant. **Clean positive byproduct:** fixed k=4 = ~85% of full SR at half the tokens, beating all adaptive variants — a strong simple baseline / analysis-paper headline. Still owed before a final verdict: (1) ~~fixed k=8 in-pipeline~~ done (0.58); (2) actual agnostic-mix at the predictor's **{1,2,8}** marginal (can't be interpolated from fixed-k — must run; note a {4,8} mix may even beat a {1,2,8} mix since k=4 is more useful per token than k=1/2); (3) 4-class predictor retest (current one skips the valid k=4 — see memory `predictor-4class-budgets`).

**KEY FINDING — the tokenizer only supports k∈{1,2,4,8} (pow2).** `train_oattok.yaml` uses `token_dropout_mode: 'pow2'` with `num_registers=8`, so `MaskedNestedDropout` trains the decoder only on `keep_k ∈ {1,2,4,8}`. k=3,5,6,7 are **untrained budgets** → degraded reconstruction. Mean offline err(k) confirms non-monotonicity: `k1 .158, k2 .138, k3 .131, k4 .129, k5 .141, k6 .142, k7 .133, k8 .124` — dips at the trained budget 4, **jumps up at 5–6**, min at 8.
- Consequence: **fixed k=5/6 are invalid baselines** (they penalize the decoder for an untrained budget, not for fewer tokens). `SR(6)=0.446 < SR(5)=0.496` is this artifact, not a real frontier. Valid fixed points are only **{1,2,4,8}**.
- This explains the predictor: its `pred_hist` is concentrated on **{1,2,8}** (trained budgets) — its "mean 6.29" is a mix of mostly-8 + some-1/2, **never actually 6**. The earlier "binary collapse to {1,2,8}" was the predictor *correctly* learning to avoid untrained budgets, not a failure.
- **Reframes the gate (revives adaptivity with an OAT-specific story):** a fixed integer budget is effectively restricted to {1,2,4,8}; to hit an intermediate *average* cost you must **mix** trained budgets per-sample, and adaptive per-obs mixing is the legitimate way. New gate baseline = the best **obs-agnostic mixture** of {1,2,4,8} at equal mean cost; the predictor wins only if obs-conditioning beats the mixing rate alone.
- **Re-run gate with valid points:** fixed k∈{1,2,4,8} (have k=8≈0.58), then obs-agnostic {1,2,4,8} mixtures at mean 5.31 / 6.29 vs predictor w2.0/w4.0.

**4-class predictor result (retrain on {1,2,4,8}, w=4.0, 2 exp):** SR ≈ 0.525 @ 6.13 tokens; in-sim k-hist ≈ {1:0.135, 2:0.14, 4:0.022, 8:0.703}. **The 4-class hypothesis is disproven:** despite giving k=4 a fair 9% of labels, the predictor still uses k=4 only ~2% in sim and is **no better (slightly worse) than 8-class** (0.559 @ 6.29). It sits **on/below the {4,8} mixing line** (0.541 at 6.13) → does not beat a simple fixed {4,8} mix. Combined with fixed k=4 dominating the low end, the K-axis gate is **solidly negative**: every adaptive variant (8-class w2/w4, 4-class w4, entropy) clusters on the fixed-k frontier, none clearly above.

**K-GATE CLOSED — NEGATIVE (agnostic-mix confirmation).** Ran the obs-agnostic mix at the 4-class marginal `{1:.137,2:.138,4:.022,8:.703}` (4 exp): **SR 0.510 ± 0.020 @ 6.13 tokens**, vs the obs-conditioned 4-class predictor **0.525 @ 6.13** (2 exp). Gap +0.015 is **within noise** → obs-conditioning adds nothing over the mixing rate. The predictor's value is only *how often* it picks k=8, not *which* obs gets which budget. Moreover both sit **below** the {4,8} interpolation line (~0.541) → the predictor's {1,2,8}-heavy marginal is likely even worse than a simple {4,8} mix. **Verdict: K-axis adaptivity is dead** (vs fixed k=4, vs agnostic-mix, and 4-class didn't rescue it). Pivot to the R-axis gate (fixed-R sweep via `--n_action_steps`) — the only remaining shot at a real (latency) improvement, since R cuts the dominant vision-CNN cost.
- **Tokenizer fork:** pow2 is a *design choice* limiting the budget to 4 levels. Retraining the tokenizer with **uniform nested dropout** (all k∈{1..8} valid) would make the err curve monotone, enable fixed k=5/6, and give granular budgets — a FASTer-style lever and possibly a separate contribution.

Conclusions:
- **Predictor scales sensibly:** w=2.0→4.0 climbs 0.497→0.559 as tokens go 5.31→6.29; **w=4.0 is the attractive operating point** (only ~0.02 SR below full for −21% tokens).
- **Learned predictor tentatively Pareto-beats entropy:** predictor line (slope ~0.063 SR/token) interpolated to 5.68 tokens ≈ 0.52 vs entropy's 0.501 at the same budget. CIs overlap at the margin → suggestive, not proven.
- **But the decisive comparison (adaptive vs fixed-k) is still missing.** All points above are vs full budget or vs the entropy heuristic (another adaptive method). Until fixed k=5/6 SR is measured, we cannot claim adaptivity beats a constant budget. This is GATE TODO #6.
- The earlier "~9pp drop is method-independent (compounding)" read still holds at the low-token end (w2.0 ≈ entropy); w=4.0 shows the drop shrinks fast as budget rises.

Speed note: eval is dominated by simulation (obs cameras rendered every step for all parallel envs). Use `MUJOCO_GL=egl` (GPU offscreen; training slurm sets it, eval did not), lower `n_test`/`n_test_vis` for iteration. (`eval_policy_sim.py` does not yet expose runner overrides via CLI.)

### Paper directions & related work

**Paper slot (strong motivation):** OAT (RSS 2026) explicitly leaves *adaptive autoregressive depth* as an open problem — it provides prefix-decodable tokens (anytime fidelity↔compute trade-off) but token count is **fixed at deployment**, and they call for a solution "grounded in uncertainty and information, rather than ad hoc engineering heuristics." Our project tackles exactly this.

**Diagnosis so far (the negative result that motivates the method):** naive adaptive budgeting does **not** beat fixed-k:
- MSE-to-demo labels are an open-loop, single-chunk **proxy** → ignore compounding error over the episode.
- The obs-only MLP predictor is **blind to the generation process** (no better than entropy) and **collapses to binary** (k∈{1,2,8}; middle k noisy/unpredictable).
- Token-entropy stopping is the "ad hoc heuristic" OAT warns against; measured in token space, not action space.
- Distribution shift: labels on demo states (mean k 4.76) vs rollout states (5.31).
- End-to-end: learned ≈ entropy in SR, both ~8–10pp below full budget → cost looks **method-independent (compounding)**.

**Related work & borrowed ideas:**
- **AAC** (Adaptive Action Chunking, CVPR 2026, [2604.04161](https://arxiv.org/abs/2604.04161)) — adapts chunk *size* (action steps) using **action-space entropy** as the cue. Borrow: use action-space uncertainty (not token entropy) as the stopping signal. Different axis (chunk size vs token count) → cite as motivation, possibly a 2nd adaptive axis.
- **SkiP** (When to Skip vs Refine, RLBench, [2605.15536](https://arxiv.org/abs/2605.15536)) — per-step binary **"refine-or-stop"** with a learned predictor + DAgger (Ross et al.) on-policy data. Borrow: reframe from "predict k from obs" to **per-step refine/stop conditioned on the partial generation** (fixes obs-blindness; binary per-step is easier than 8-way min_k).
- **LAC** ([2602.00686](https://arxiv.org/pdf/2602.00686)) — learnable visual-token caching trained on **task-loss feedback**; 2–3× on LIBERO without SR loss. Borrow: ground the controller's signal in task outcome, not a reconstruction proxy. (LAC cuts *visual* tokens at the encoder; we cut *action* tokens at generation → orthogonal, combinable.)
- **Spec-VLA** ([2507.22424](https://arxiv.org/abs/2507.22424)) — speculative decoding for VLAs; orthogonal lossless speedup, combinable.
- **FASTer** ([2512.04952](https://arxiv.org/abs/2512.04952)), **FAST** ([2501.09747](https://arxiv.org/abs/2501.09747)) — learnable / DCT tokenizers; relevant if we instead retrain the tokenizer so fewer tokens suffice (attacks compounding ceiling directly).

**Proposed method (synthesis):** a **generation-aware, per-step "refine-or-stop" controller** for OAT's ordered tokens, using an **action-space uncertainty signal** (action convergence `Δ(a_k, a_{k−1})` or action entropy), with the **decision threshold/head trained on task-grounded, on-policy (DAgger) data** rather than MSE-to-demo. Realizes OAT's "uncertainty/information" desideratum. Training the signal: prefer **offline logged-bandit** (`P(success | features, k)` from fixed-k rollouts — partly produced by the `SR(k)` sweep) over fragile online RL; escalate to DAgger iterations only if off-policy mismatch bites.

**Novelty positioning:** first adaptive-depth controller for *prefix-decodable / ordered action tokenizers* (anytime fidelity), + the diagnosis of why naive variants fail. Narrower than it looks — AAC already does action-entropy adaptivity for VLA (chunk size), so position carefully on the token-count axis + ordered-tokenizer specificity + the failure analysis.

**Two-axis extension — joint `(k, L)` control:** there are two orthogonal adaptive axes, controlling *different costs*:
- **Axis 1 — token count `k`** (fidelity of the chunk). Cost = AR generation, which is **cheap** (a few small transformer steps). This is OAT's anytime axis.
- **Axis 2 — executed chunk length `L`** (`n_action_steps`, currently 16/32). Controls **how often we replan**, and each replan pays the **expensive vision CNN** (2 cameras). Episode cost ≈ `(episode_len / L) × (CNN + k·AR)`, so longer `L` cuts the dominant cost. This is AAC's axis.

Implications: (a) the token-count axis we've optimized is the *cheap* one — `measure_latency_adaptive` at batch=1 (TODO #10) must confirm whether reducing `k` saves meaningful wall-clock at all, or whether `L` is the higher-leverage axis. (b) The axes are **coupled, not independent**: a long open-loop `L` demands high fidelity → high `k`; short `L` tolerates low `k` (replans soon). Compounding error accumulates *within* the open-loop segment, so cutting `k` and extending `L` together is risky. → a **joint/coupled controller** `(k, L)` (one uncertainty signal drives both, or pick `L` by reactivity then `k` by the fidelity `L` needs), **not** two independent predictors. (c) Both axes are available on the current OAT model without retraining (decode horizon is 32; execute `L∈[1,32]`, generate `k∈[1,8]`). (d) Novelty caveat: axis `L` alone = re-doing AAC; the defensible contribution is the **joint `(k,L)` on a prefix-decodable tokenizer + the fidelity↔length coupling**. Sequencing: gate (#6) first, then a latency measurement to decide which axis is worth it, before building the joint controller.

**Caveat on per-obs credit assignment:** task-grounded labels (B/C) inherit a fundamental difficulty — episode success is binary over ~34 chunk decisions, so per-obs attribution is noisy (mitigate by per-episode labels, counterfactual single-step k variation, or averaging over many rollouts).

### Proposed method — Closed-loop Adaptive OAT (`(K, R)` controller)

Per replan, a lightweight controller picks a pair `(K, R)` from obs features: **`K`** = number of OAT tokens to generate (autoregressive depth), **`R`** = number of leading continuous actions of the decoded chunk to execute open-loop before the next replan. OAT tokenizer/decoder/policy stay **frozen** (it still decodes the full fixed-length chunk); we only control refinement depth + open-loop horizon.

- **Data:** rollout-state dataset (not just demos) — states visited by full-OAT8, fixed-k, the current predictor, and *failed* rollouts.
- **Labels (full-OAT8 teacher):** for each `K`, prefix-decode the chunk; for each `R`, check **closed-loop consistency** — do the next `R` actions of the prefix-`K` chunk match what a fresh full-OAT8 replan would produce in the corresponding future states. Target `(K,R)` = cheapest pair preserving sufficient consistency, with `cost(K,R) = (C_vision + C_AR·K) / R` (vision-CNN per replan amortized over executed steps).
- **Training:** small pair-policy `π(K,R | obs_features)` via supervised / soft offline-bandit target; optional online refinement with reward `success − λ·tokens − μ·num_policy_calls − η·jerk`, OAT frozen.
- **Goal:** *approach* full-OAT8 success at lower total inference cost (fewer action tokens, fewer replans, fewer vision-encoder passes) → better SR-vs-latency Pareto. Not to beat OAT8.

**Why it's the right synthesis:** joint `(K,R)` on a frozen prefix-decodable tokenizer (only OAT gives the `K` axis); the **amortized cost objective** is the real Pareto target and is absent in AAC; closed-loop teacher labels are cheaper than sim-success.

**Risks / gates (must address before trusting it):**
1. **Not yet gated on headroom.** Our k-gate is negative (adaptive K ≈ fixed K at ~5 tokens); the R axis is untested. → run a **2D fixed-`(K,R)` sweep first**; the learned controller must beat the best *constant* `(K,R)`.
2. **Offline target is still a per-chunk proxy** (consistency<ε ≠ task success) — the same trap as MSE `min_k` (offline-good ≠ SR-good, cf. w=2.0). So offline = **prior only**; the real work falls on the **fragile online RL**. Start with offline logged-bandit; online only if off-policy gap bites.
3. **Cost gradient pushes `(small K, large R)`** — `C_AR·K` is small vs `C_vision`, and dividing by `R` rewards long open-loop execution of *coarse* (small-K) chunks = max compounding. Needs a guard (min fidelity per `R`, or constraint `K ≥ f(R)`).
4. **`R` label is off-policy** — future states depend on what was actually executed (prefix-`K`), not on the OAT8 trajectory; label along the candidate's own short rollout or accept approximation + DAgger.
5. **Value may concentrate in `R`** (K cheap + low-headroom, and cost barely depends on K) → risks collapsing to "R matters, K≈const" = AAC. Defend novelty via joint+cost+ordered-tokenizer; **include an AAC-style action-entropy `R` baseline**.
6. **Strong distribution shift from `R`** (changes replan schedule → visited states) → **DAgger** iterations required, not optional.

### R-axis (chunk length) gate — PROMISING (opposite of K)

Fixed-R sweep at K=8 (full budget), via `--n_action_steps R --entropy_threshold 0 --use_k_tokens 8` (2 exp each):

| R | SR | replan cost |
|---|-----|-------------|
| 8 | **0.635** | 2× |
| 16 | 0.577 | 1× (OAT default, ≈ paper 0.58 — sanity ✓) |
| 24 | 0.510 | 0.67× |
| 32 | 0.440 | 0.5× |

- **SR rises steeply as R shrinks**: R=8 (0.635) **beats** R=16 (0.577) and the paper's 0.58 → replanning more often gives more success; OAT's default R=16 leaves success on the table. **But it's a trade-off, not free** — R=8 doubles the (dominant) vision-CNN cost.
- **R is high-leverage** (steep curve) — unlike the flat/dead K axis → there is something to exploit.
- **The SR(R) curve is clean/real, NOT an artifact** (unlike pow2 k=5,6). Steps 17–32 *are* trained — the tokenizer reconstructs all 32 (MSE over 32) and the 8 tokens jointly encode the full 32-step chunk. So R∈[1,32] are all valid operating points. The drop at large R is genuine **reactivity loss + compounding + inherent far-future-prediction difficulty** (predicting 24–32 steps ahead from one obs), not under-training. Only caveat: OAT was designed/reported at R=16, so R=24,32 is a worse point on a real trade-off, outside their chosen operating regime.
- **Joint (K,R) motivation weakened:** "high K enables long R" fails because long R is bad *regardless* of K → the real lever is **adaptive R at K=8** (closer to AAC — novelty caveat).
- **Next:** (1) add R=4 to find the SR(R) peak; (2) build variable-R execution (policy returns an R-length chunk, runner executes it, replan) + a heuristic R signal (fidelity-ladder agreement) → test whether **adaptive R beats fixed R at matched mean cost** (the real R-gate: needs per-obs heterogeneity). Steep curve ⇒ potential payoff is large if heterogeneity exists.

### GATE 1 implementation — variable-R execution (DONE, ready to run)

Per-observation adaptive executed-chunk-length R, **K held fixed at full budget (8)** to isolate the R axis (directly comparable to the fixed-R sweep). Mechanism: policy returns the full r_max-length chunk + per-sample `r_exec`; runner NaN-pads each env's chunk beyond its R; `MultiStepWrapper` stops at the first all-NaN row. Envs then desync in sim-time (independent episodes).

- **`oat/oat/policy/oatpolicy.py` — `predict_action_variable_r`** (dispatched from `predict_action_adaptive` via the `adaptive_r` kwarg, taking precedence over K-budget modes). Generates one full-budget chunk (KV-cache), then picks R per obs:
  - `convergence` (generation-aware R-signal): decode the SAME tokens at `r_coarse_k` (=4) and at K; per-timestep divergence `d_t = ||norm(A8)_t − norm(A4)_t||` (normalizer space); R = length of the leading prefix where coarse & fine plans agree (`d_t < r_threshold`), clamped `[r_min, r_max]`; early divergence → short R. Reads the decoded plan, **not** obs → dodges the K-gate's obs-wall.
  - `random` (control): R ~ Uniform{r_min..r_max} — variance in R uncorrelated with obs; must NOT beat fixed-R at matched mean if obs-conditioning is the source of any gain.
  - `fixed` (sanity): R = r_max for all (== fixed-R sweep at R=r_max).
  - Returns `{action (full r_max chunk), r_exec [B], n_tokens (=K), div_mean}`.
- **`oat/oat/gymnasium_util/multistep_wrapper.py`** — `step` breaks on an all-NaN action row (the variable-R sentinel).
- **`oat/oat/env_runner/libero_runner.py`** — NaN-pads each env's chunk beyond `r_exec`; finite-check only the executed prefix; `pbar.update(min(r_exec))` so the slowest (small-R) env is never cut short (provable: `pbar.n = Σ min_i r_exec ≤ min_i Σ r_exec` = slowest env's accumulated steps); reports `mean_r_exec` (= mean replan interval; replan/vision-CNN cost ∝ `max_episode_steps / mean_r_exec`) + an R histogram.
  - **Desync correctness fix (`episode_done` per-init):** variable-R desyncs the envs, so a fast (small-R) env that *fails* can truncate early, autoreset, and run a 2nd episode while slow envs still run (`AsyncVectorEnv` autoresets on done). Without a guard, a counted 2nd-episode success would **inflate** adaptive-R SR (fixed-R never hits this — lockstep truncation ends the loop before autoreset). Fix: freeze each env after its FIRST episode; success counted only in the first episode; loop ends when all first episodes are done. Reduces to the original lockstep behaviour for fixed-R. (NB: `n_chunks = ceil(n_inits/n_envs)` is the env-batch count, NOT a time/episode-length bound — episode length is the inner `while pbar.n < max_episode_steps` + per-env wrapper truncation, so it is R-independent.)
- **`oat/scripts/eval_policy_sim.py`** — flags `--adaptive_r {convergence,random,fixed}`, `--r_coarse_k`, `--r_min`, `--r_max`, `--r_threshold`; forces `n_action_steps = r_max`; prints `mean R`.

Run (sanity first — variable-R plumbing must reproduce fixed R=16 ≈ 0.577):
```bash
cd oat && MUJOCO_GL=egl uv run scripts/eval_policy_sim.py \
  -c my_models/policy_ep-0250_sr-0.596.ckpt -o eval_out/varR_sanity \
  -n 2 --use_k_tokens 8 --adaptive_r fixed --r_min 16 --r_max 16
# then: --adaptive_r convergence --r_min 8 --r_max 32 --r_threshold {0.2,0.3,0.5}  (tune so mean R ~16)
#       --adaptive_r random --r_min 8 --r_max 32   (control)
```
**Gate read:** `convergence` passes only if SR(adaptive @ mean R̄) is **above** the fixed-R sweep interpolation at R̄ **and** above `random` at the same mean. If convergence ≈ random ≈ fixed-curve → no R-heterogeneity from this signal (K-axis déjà vu) → the whole patch→R coupling is moot. **Likely need to sweep `--r_threshold`** (0.5 in norm-L2 over 7 dims may rarely trigger → mean R → r_max). **Possible v2 signal: an oracle R** (offline teacher-consistency) to answer "is there ANY R-heterogeneity" before blaming the cheap signal.

### GATE 1 — progress (2026-06-04): horizon bug fixed, offline divergence diagnosed, sim pending

**BUG FOUND & FIXED (was silently breaking variable-R):** `predict_action_variable_r` resolved `r_max` against `action_tokenizer.latent_horizon`, but that is **8 = the token-register count (num_registers)**, NOT the action horizon. The decoded chunk is **32 steps** (`tokenizer.py:97` asserts `eval_keep_k ≤ latent_horizon`, i.e. latent_horizon caps the *token budget* k∈[1,8], not time). So `--r_max 32` silently clamped to 8 → `r_min=r_max=8` → `r_exec=8` for every sample regardless of the signal. The first convergence sim run (`hist(R=8..8)`, SR≈0.605 = fixed-R=8) was this artifact, **not** a property of the signal — it must be re-run.
- Fix: resolve `H = action_pred.shape[1]` (decoded action length, =32) AFTER detokenize, in both `oat/policy/oatpolicy.py:predict_action_variable_r` and the new diagnostic `oat/scripts/diag_convergence_div.py`.

**Offline divergence diagnostic** (`oat/scripts/diag_convergence_div.py`, NO sim): decodes coarse(k=4) vs full(k=8) on real dataset obs, dumps per-timestep `d_t = ||norm(A8)_t − norm(A4)_t||` percentiles + a threshold→R-distribution sweep. Run:
```bash
cd oat && uv run python scripts/diag_convergence_div.py \
  -c my_models/policy_ep-0250_sr-0.596.ckpt --max_samples 4096 --r_max 32 [--r_coarse_k 2]
```
Result (N=8800, coarse_k=4):
- **`d_t` is TINY and FLAT across all 32 steps** (p50 ~0.05–0.11, p90 ~0.12–0.20, p99 ~0.13–0.33); far-horizon steps 24–27 are even the *lowest* (p50≈0.05), not higher. coarse k=4 ≈ full k=8 **everywhere** (consistent with offline err k4=.129≈k8=.124).
- **Two consequences:** (a) no within-chunk localization structure → the "first threshold crossing" mechanism mostly picks up each sample's *overall* divergence magnitude + noise → the signal collapses to a **per-sample scalar**, not "where to refine"; (b) it directly undercuts the **sparse-patch leg of H-OAT**: residual targets `decode8−decode4` are ~0.05 → almost nothing to patch.
- **Threshold→R sweep DOES produce a spread** (not degenerate): thr 0.10→meanR 13.7 std 8.99 (60%@8, 12%@32); thr **0.12 ≈ meanR 16** target; thr 0.15→meanR 22.3 std 10.75. Both ends populated → bimodal. So heterogeneity *in the signal* exists; whether it's *useful* (vs noise) needs sim.

**coarse_k=2 is a BETTER signal than coarse_k=4 (N=15008) → use k=2 for the sim.** Two gains: (a) `d_t` is **no longer flat** — a real within-chunk trend appears (early steps lower: t0–7 p50≈0.12–0.13; far horizon higher: t24–31 p50≈0.15–0.17, p99≈0.5–0.6), so "first crossing" picks up genuine early-confident→late-uncertain structure, not pure noise; (b) ~1.5× more dynamic range (p50 0.12–0.17 vs 0.05–0.11). Spread is nicely bimodal (thr 0.2: hist [5956,1334,1698,6020], 30%@8/30%@32). For mean R≈16 use **thr≈0.17** (offline meanR 14.0@0.15, 19.9@0.2). Caveat: the early-low/late-high trend is **generic** across samples (reflects "k=2 reconstructs the far future worse" — a tokenizer property), so the per-obs-useful part is still mostly the per-sample divergence *magnitude*; sim decides if that's useful. → **queued sim switched to `--r_coarse_k 2 --r_threshold 0.17`.**

**Decisive sim pair queued (READY, not yet run):** convergence @ thr 0.12 (mean R~16) vs random control @ mean 16, n=3 each:
```bash
# convergence (coarse_k=2 — better-conditioned signal than k=4)
MUJOCO_GL=egl uv run scripts/eval_policy_sim.py -c my_models/policy_ep-0250_sr-0.596.ckpt \
  -o eval_out/varR_conv_k2 -n 3 --use_k_tokens 8 --adaptive_r convergence \
  --r_min 8 --r_max 32 --r_coarse_k 2 --r_threshold 0.17
# random @ mean 16
MUJOCO_GL=egl uv run scripts/eval_policy_sim.py -c my_models/policy_ep-0250_sr-0.596.ckpt \
  -o eval_out/varR_rand16 -n 3 --use_k_tokens 8 --adaptive_r random --r_min 8 --r_max 24
```
- **Anchor / bar:** fixed R=16 = **0.577**. SR(R) is monotone-decreasing & mildly **concave** (0.635/0.577/0.510/0.440 @ R=8/16/24/32) → by Jensen a constant beats a random spread, so the real bar is **convergence > fixed-16 (0.577)**; random is the secondary obs-agnostic-spread control.
- **Read:** convergence >0.577 & >random → R-adaptivity real, build on. convergence ≈ random ≈ ≤0.577 → spread is noise (expected from flat `d_t`) → this signal dead.
- **On submitting results:** report the in-sim `mean R` from the convergence log; if it drifts off 16 (rollout≠demo states), retune `--r_threshold` + random range to re-match the mean (else the cost is not matched).

**Honest prior (H-OAT overall): LOW.** K-axis closed-negative; the sparse-patch leg inherits that null PLUS tiny residuals (~0.05) PLUS Step-0 wrong-semantics. The only live bet is **adaptive R**, and its one cheap signal (convergence) looks like noise (flat `d_t`). Every adaptive variant tried so far (all K-axis) lands on the fixed frontier → structural read: this frozen OAT+LIBERO has little *readable* per-obs heterogeneity. **Next decisive test if convergence-vs-random fails = ORACLE R** (offline teacher-consistency: smallest R whose continuation matches a fresh full-OAT8 replan): if even oracle R can't beat fixed R at matched mean, no signal will → R-adaptivity (and H-OAT) is dead → pivot to the negative-result/diagnosis paper (already the stated fallback). Oracle-R is **not yet implemented** — build it only if the convergence sim fails.

### GATE 1 RESULT (2026-06-05): convergence-R is DEAD — worse than random. R-axis (this signal) CLOSED-NEGATIVE.

Decisive sim done (n=3, coarse_k=2, thr 0.17, matched mean R≈14):
| mode | SR (mean ± std) | mean R |
|------|-----------------|--------|
| **convergence** | **0.553 ± 0.036** (.578/.512/.568) | 14.1 |
| **random** (control) | **0.595 ± 0.019** (.616/.592/.578) | 14.0 |
| fixed-R interp @14 | ~0.592 | 14 |

- **convergence LOSES to random by +0.042 at matched mean** (~1.8σ; combined stderr ≈0.024) AND loses to the fixed-R curve. The obs/plan-conditioned signal is **anti-informative**, not just neutral — conditioning R on convergence is *worse* than a dumb uniform spread.
- **random ≈ fixed** @ mean14 (0.595 ≈ 0.592) → SR(R) is ~linear in [8,16] → **no Jensen room**; spreading R is neutral, so any adaptive win must come from genuine positive obs↔R-tolerance correlation. convergence's is ≤0.
- **Per-exp noise is ~0.05** (same config gave 0.528 then 0.578) → never trust 1 exp here; needs n≥3.
- Caveat (doesn't rescue it): convergence range [8,32] (bimodal 52%@8/12%@32) vs random [8,20] (uniform) — convergence's 12% mass at R=32 (worst fixed, 0.440) drags it, but if those R=32 picks were the *right* states it'd still win. Cleanest control (if ever needed) = **permutation**: shuffle convergence's own r_exec across episodes (same histogram, kills obs-correlation).
- **Ties to the central insight:** convergence = decode-k2-vs-k8 = a **reconstruction** signal → 3rd reconstruction-based negative (after min-k, entropy). `reconstruction ≠ value` confirmed again.

**Verdict:** convergence-R signal dead. R-adaptivity not *formally* closed (oracle-R = ceiling test still unbuilt), BUT given (a) convergence < random, (b) random ≈ fixed (no Jensen room), (c) the reconstruction-signal pattern → prior on R-adaptivity is now LOW. **Decision: stop chasing reconstruction R-signals; PIVOT to Value-Guided OAT** (see backlog). Build the counterfactual-sim harness once → use for oracle value(k|phase) (headline C1) + oracle-BoN (gate #7/#9) + oracle-R as a quick add-on to formally close R for the negative-results section.

### Step 0 (intra-chunk reconstruction) + H-OAT direction

**Step 0 premise check** (`scripts/per_timestep_recon_error.py`): is the *marginal value of tokens* uneven WITHIN a chunk? Autoencode GT chunks through the frozen tokenizer at k∈{1,2,4,8}; measure per-timestep error e_t(k) and the **gain** `gain_t(a→b)=max(e_t(a)−e_t(b),0)` / **excess** of extra tokens (raw error is misleading — a step can be hard at every k); per-chunk concentration (CV/peak2mean/topN), full + executed windows; + within-chunk correlation of gain with gripper-change/jerk/action-delta. (NB measure within-chunk unevenness, not the avg profile, which smears peaks across arbitrary window starts.)
- **Concentration: REAL & robust.** `gain_4to8` top2 vs uniform ref: 1.3× (R=4), 2.0× (R=8), **2.6× (R=16)**, 3.4× (full); median ≈ mean → not driven by a few extreme chunks. A few timesteps carry most of the 4→8 improvement.
- **Semantics: WEAK / FAILED.** within-chunk corr of `gain_4to8` with gripper r=+0.002 (gripper-change is *below* avg at the top-gain step, 0.52×), jerk r=+0.053 (1.22×), delta r=+0.051 (1.34×). → hard-to-reconstruct steps are **fast-motion, NOT grasp/contact**. Reconstruction-hardness ≠ task-importance (likely anti-correlated: grasp is task-critical but *easy* to tokenize). ⇒ allocating tokens by reconstruction-residual risks fixing the wrong steps → reconstruction gain may not transfer to SR.

**Chosen scheme — Generation-aware Sparse Residual H-OAT + adaptive R** (concretizes Closed-loop Adaptive OAT with a real mechanism):
```
obs → OAT coarse prefix (k=4) → decode A4
→ generation-aware router reads the COARSE PLAN (not obs): decode_k2-vs-k4 instability,
  curvature, AR hidden/logits → picks sparse refinement patches
→ patch decoder applies residuals: A_final[t:t+L] = A4[t:t+L] + ΔA_patch  (targets decode8−decode4)
→ patch activity (count/strength/position) → adaptive R: unstable→short R, stable→long R
→ execute first R actions, replan
```
- #1 SparsePatch = architecture (coarse OAT4 backbone + sparse residual patches; cheap residual targets, no full variable-segment tokenizer). #2 generation-aware router = WHERE to patch, reading the generated plan **not obs** — dodges the obs-wall that sank the K-predictor (K-gate tested obs-only, so its null does NOT apply here). One shared uncertainty signal (patch activity) drives both K (refinement) and R (horizon).
- **Where it wins (honest):** the K/patch part alone is the *cheap* axis (small latency) + threatened by the Step-0 semantic finding. **The adaptive-R coupling gives it teeth** — R cuts the dominant vision-CNN cost and has a real signal (R=8 > OAT8 SR). Win shifts from "fewer tokens" to "fewer replans / better SR-vs-cost Pareto". Cannot beat OAT8 SR (frozen policy); target = match SR at lower total inference cost.

**Gates (cheapest/most-decisive first — do NOT build the full scheme before these):**
1. **GATE 1 — R-heterogeneity (prerequisite).** Does *any* adaptive R beat fixed R at matched mean replan cost? R-sweep shows SR monotone-decreasing in R (8>16>24>32), so "confident→long R" helps only if some chunks tolerate long R. Test with variable-R execution + a simple signal (action-convergence `Δ(a_k,a_{k-1})` or oracle) **before** building patches. If no headroom → the whole patch→R coupling is moot.
2. **GATE 2 — patch-signal vs simpler R signals** (action-entropy / AAC-style): does the generation-aware patch signal beat them? Else it reinvents AAC.
3. **GATE 3 — SR-translation + semantics.** Patches target fast-motion (Step 0); does the scheme maintain SR, and does "patch activity → R" align with task needs?

**Immediate cheap step:** GATE 1 — variable-R execution in the runner + a simple R signal, compare adaptive-R vs fixed-R on SR-vs-replan-cost. Decides the fate of the whole joint idea before any patch machinery.

### Idea backlog — if R-axis fails: Test-time scaling for OAT (best-of-N over ordered tokens)

**Pivot target if GATE 1 (R) closes negative.** Stronger bet than adaptivity: adaptivity is *capped by the policy's own SR* (best case = approach full-OAT8 cheaper); test-time selection can **exceed** it. Grounded in our own measurements + a hot, *positive*-result literature (vs the thin adaptivity literature).

**Measurement grounding (all ours):** (1) tokens are redundant — k=2≈k=4≈k=8 (err .129/.124) → generation is cheap & the "anytime fidelity" range is near-empty (this is WHY K-adaptivity died); (2) AR is cheap, **vision-CNN dominates** (obs_encoder 22.4M ≫ policy 5M), cost ≈ `(C_vision + C_AR·K)/R`; (3) adaptivity (K and R) is capped by the frozen policy's SR. ⇒ redirect the cheap, redundant generation budget from "spend less" (adaptivity, no headroom) to "spend the same on **selection**" (best-of-N), which can beat the base policy.

**Method — shared-perception best-of-N + prefix token-tree search.** Per replan: encode vision **once** → sample **N** candidate token-chunks from the cheap AR head (all conditioned on the same features) → score → execute the best. Optional tree-search: branch on the **first 1–2 tokens** (carry ~95% of the chunk per k=2≈k=8), decode the rest cheaply, prune with the scorer.
- **Why OAT is the ideal substrate (architectural win, measurement-backed):** the dominant cost (vision) is computed **once and amortized across all N candidates** → best-of-N is nearly free on the expensive axis. RoboMonkey/diffusion VLAs lack this clean perception↔generation split (their sampling is the expensive part). Plus redundant/ordered tokens → cheap sampling + efficient prefix tree-search.
- **Anti-compounding:** best-of-N lowers per-chunk error → less compounding over the open-loop horizon (the very thing the R-sweep exposed).
- **Refinement — EXHAUSTIVE prefix enumeration (not random best-of-N).** Since the first ~2 tokens set the bulk of the chunk (k2 within ~0.05/dim of k8; refinement from tokens 3–8 is marginal) and `topk` restricts each token to ~10 options, the *meaningful* candidate space is only **top-M(token1) × top-M(token2) ≈ 100 chunks**. So **enumerate** that low-dim, high-value prefix grid, decode each (k=2 decode ≈ k=8), greedy-fill the near-irrelevant tail 3–8, score all ~100, execute best — strictly better coverage than random sampling N, still cheap (vision shared, AR trivial, scorer = small MLP ×100). "Spend more compute on the first 2 tokens" on a *frozen* model can only mean **search/selection over their output** (can't make the frozen generator smarter) → it collapses to this prefix enumeration, NOT a smarter generator.
  - Caveat: this is an *allocation/efficiency* win on top of best-of-N — it does **not** bypass GATE A (headroom) or the scorer. If oracle-best-of-prefix doesn't beat greedy, no compute allocation rescues it. Accuracy note: "2 tokens ≈ 95%" is loose — precisely A2 differs from A8 by ~0.05/dim (`d_t`≈0.13 over 7 dims), err .138→.124 k2→k8; 2 tokens set the *shape*, 3–8 refine.
  - **Cheap diagnostic to fix the search shape (do BEFORE building, like `diag_convergence_div`):** sample N token-seqs on shared features, decode, measure what fraction of the chunk's **action-space variance** is explained by token positions 1–2 vs 3–8. Variance dominated by 1–2 → prefix enumeration covers the whole meaningful space (BoN becomes near-oracle on coverage; only the scorer is left). Variance spread across positions → staged beam (expand+select per 2-token block) regains value. NB reconstruction-redundancy (k2≈k8, decoder property) ≠ sampling-diversity (policy property) — this diagnostic separates them.
  - **Heavy-prefix retrain variant (separate, non-frozen branch):** "more compute on 2 tokens" could instead mean an asymmetric **heavy-prefix / light-tail** architecture (a stronger head distilled to predict the first 1–2 tokens better) or a tokenizer fork (uniform nested dropout so all 8 tokens carry info). Requires retraining → distinct contribution from the frozen inference-time story.

**Scoring, escalating (cheapest first):**
- (free) **verifier-free**: rank N by intrinsic confidence — likelihood / low token-entropy (already computed) / action-space majority-vote. MG-Select-style.
- (trained) **learned action-value verifier** on rollouts we already generate (success-labeled, MC returns for credit assignment). RoboMonkey/RoVer-style.

**Cheap gates (same oracle discipline as oracle-R — test EXISTENCE before building):**
- **GATE A (oracle best-of-N, decisive, ~an evening):** sample N, pick the candidate that *actually* succeeds (counterfactual sim) or best-tracks the demo. oracle-BoN ≫ single → diversity + headroom exist → build the verifier. oracle ≈ single → policy samples too similar → raise temperature / Gaussian-perturb the FSQ codes (RoboMonkey trick adapted to discrete latents), else dead.
- **GATE B (verifier-free, free):** does intrinsic-confidence ranking already beat single-sample? If yes, paper without training a verifier.

**Novelty positioning:** *first test-time scaling for ordered / prefix-decodable discrete action tokenizers.* Wedges vs the field: (1) ordered discrete tokens (vs continuous/diffusion in RoboMonkey/MG-Select/RoVer); (2) **prefix token-tree search** with early-token branching (unique to OAT, justified by k=2≈k=8); (3) **perception amortized once across N** (architectural efficiency RoboMonkey lacks); (4) motivated by our K/R-adaptivity negative diagnosis.

**Related work:** RoboMonkey ([2506.17811](https://arxiv.org/abs/2506.17811), sample+Gaussian+VLM-verifier, inference scaling law, OpenVLA 49.8→56.5); MG-Select / verifier-free ([2510.05681](https://arxiv.org/pdf/2510.05681), ICLR 2026, intrinsic-confidence BoN); RoVer ([2510.10975](https://arxiv.org/pdf/2510.10975), reward-model verifier). Cheap-replan fallback (recovers but can't exceed SR): VLA-Cache ([2502.02175](https://arxiv.org/abs/2502.02175), NeurIPS 2025), LAC ([2602.00686](https://arxiv.org/pdf/2602.00686)). Real-time chunking: RTC ([2506.07339](https://arxiv.org/abs/2506.07339)).

**Risks:** (a) OAT sample diversity may be low (temp=1/topk=10) → GATE A tests this first, cheaply; fix via temperature / FSQ-latent perturbation. (b) verifier credit assignment (recurring trap) → verifier-free start + MC returns; field has positive results, lower risk than adaptivity.

**Paper structure:** §diagnosis (anytime tokens ≠ anytime inference; K dead via redundancy; R [gate]; vision-dominated cost — mostly DONE) → §insight (redirect cheap generation to selection; perception amortized) → §method (shared-perception BoN + prefix tree-search + verifier free→learned) → §results (SR vs N scaling law; SR vs latency; vs base policy; vs all adaptivity baselines).

### Idea backlog — Prefix-Guided Visual OAT (attack the VISION axis, NOT another action-token method)

**Strongest-targeted idea: hit perception, the actual bottleneck.** Our diagnosis says: K-axis dead (action tokens redundant), AR cheap, **vision-CNN dominates** (22.4M ≫ 5M), reconstruction-adaptivity ≠ SR. ⇒ real leverage is *what the model sees*, not the action-token budget. Use the cheap **coarse OAT prefix as a query into vision**: generate `z1,z2` → decode `A2` coarse intent → build action-queries from `A2` (+ proprio + token hidden + uncertainty) → select visual tokens/regions/cameras for the precise action → either (A) regenerate `z3..z8` from the action-relevant visual context, or (B) reselect visual context for full `z1..z8`. The two LIBERO cameras give a clean gating substrate: **agent-view** (approach / global goal) vs **eye-in-hand** (contact / local gripper-object detail). `gate=σ(MLP(q)); ctx = gate·eye + (1-gate)·agent`.

**This is NOT frozen-inference — it RETRAINS, and that is a PLUS, not a minus** (corrected stance, user pushback accepted): a co-trained action-prefix-conditioned visual selector is a genuine *architectural* contribution (more novelty than a frozen inference trick); baseline becomes "vanilla OAT, same training budget" vs this (fair, even cleaner than the frozen anchor); retraining *opens* the design space (Variant A no longer has to preserve the frozen AR head's input dist; can co-train tokenizer / nested-dropout too). The earlier "scope jump = con" is **retracted**.

**Generation-aware → dodges the obs-wall:** conditioning visual selection on the *decoded coarse plan* `A2` (not raw obs) means the K-predictor's obs-only null does NOT apply here.

**Two genuine residual cautions (NOT anti-retrain — orthogonal):**
1. **Gate the PREMISE before the (expensive) retrain.** Cheap insurance, not an argument against training. **`phase × camera masking` on the FROZEN policy (~an evening):** mask one camera as a function of plan phase (gripper-open/approach vs close/contact from `A2`), measure SR. Mask agent-view during contact → SR holds, eye-in-hand during contact → SR drops (and vice versa for approach) ⇒ phase-dependent camera importance is REAL → headroom → build the selector. SR insensitive to which-camera-when ⇒ premise dead, skip the retrain. This is the oracle-style existence gate (analog of oracle-R / oracle-BoN), and it's independently publishable as a diagnosis ("camera importance is/ isn't phase-dependent in OAT-LIBERO").
2. **Decide Cost vs Quality (retrain doesn't resolve this — orthogonal design choice).** SR↑ → dense cross-attention selector (LightVLA-style regularization); latency↓ → **conditional encoding** (skip the 2nd camera / fine-resolution unless the coarse plan flags imminent contact) — this is the version that actually cuts the dominant vision cost (post-hoc attention over already-encoded tokens does NOT save encoder FLOPs); or Pareto (both). Picking this sets the mechanism AND the baselines.

**Semantic risk (Step 0):** reconstruction-hardness ≈ fast-motion, NOT grasp/contact → plan-derived queries may pick visual regions by *motion* not *task-criticality* (same trap as the patch idea). The gripper channel of `A2` (dim 7, flags imminent close) is the concrete testable cue to ground selection in task-semantics.

**Sharpened cost-version (recommended if premise-gate passes):** "**coarse-action-plan-gated conditional perception invocation**" — mostly run cheap/single-camera perception, invoke the expensive 2nd camera / fine-resolution only when `A2` flags imminent contact. Cuts the dominant cost (grounded in our bottleneck finding), cleaner than a dense selector, gated by the same phase×camera masking, novelty = plan-gated conditional perception on a prefix-decodable tokenizer.

**Related work (live, supportive, but CROWDED → wedge must be sharp):** VLA-Pruner (dual-level visual importance: semantic attention + action-decode attention — already "action-aware"; our wedge = explicit *coarse-prefix-as-plan* pre-selection, two-stage, vs their during-decode attention — must show better/cheaper, else "VLA-Pruner on OAT"); LightVLA (learnable visual pruning ↓FLOPs AND ↑SR — pruning as regularizer); Compressor-VLA (holistic task ctx + fine-grained spatial via two modules ≈ our coarse-plan + local-refinement split). Connects to BLT/H-Net: content/context-dependent allocation, moved into the *visual* axis instead of the action chunk.

**Priority vs test-time BoN:** BoN is cheaper, frozen, has a clean oracle gate, and can exceed the SR ceiling → keep it FIRST. But the `phase×camera masking` gate here is so cheap it's worth running in parallel — independently valuable as a diagnosis regardless of whether the full selector gets built.

### Idea backlog — Value-Guided OAT (the CENTRAL insight: stop measuring reconstruction, measure VALUE)

**THE unifying lesson from a ~30-paper survey (2025–26).** Every one of OUR negatives used a **reconstruction / heuristic** signal; every POSITIVE result in the field uses a **value / reward** signal:
| our NEGATIVES | signal | | field POSITIVES | signal |
|---|---|---|---|---|
| min-k predictor | MSE-to-demo (reconstruction) | | V-GPS, VGAS | offline-RL value |
| convergence-R | decode-k4-vs-k8 (reconstruction) | | Adaptive Q-Chunking | RL advantage |
| entropy-stop | token entropy (heuristic) | | RoboMonkey / TACO | reward / pseudo-count verifier |
We kept measuring "how close is the chunk to the demo"; we should measure "how much does this chunk lead to success." **Reconstruction-fidelity ≠ task-value** (Step-0 confirmed: hard-to-reconstruct = fast-motion, not grasp). This single reframe likely explains ALL the K/R negatives — the *mechanisms* may be fine, the *label* was wrong.

**Method.** Train a value/Q function `V(features, chunk)` (or token-level `Q`) via offline RL on rollout data (Q-chunking recipe [2507.07969](https://arxiv.org/abs/2507.07969) / V-GPS [2410.13816](https://arxiv.org/html/2410.13816v2)). Use it two ways:
1. **Test-time re-ranking → SR↑ ABOVE BC.** Generate N candidates (vision amortized, cheap), rank by value, execute the best. V-GPS/VGAS show this *exceeds* the base policy (RL-value improves over demos) — adaptivity never could (capped at BC).
2. **Value-grounded adaptive depth/horizon → REVIVE K/R.** Re-run the K-budget and R-horizon choice driven by value-advantage (à la AQC) instead of reconstruction. K/R adaptivity may not be dead — the reconstruction *label* was.

**Why OAT is the unique substrate (wedge vs V-GPS/VGAS/RoboMonkey, which re-rank only WHOLE actions):**
- **Prefix-decodable → token/prefix-level value (process reward).** Any prefix decodes to a full chunk → evaluate value on a *partial* generation and **steer generation token-by-token** (value-guided decoding, reasoning-LLM style). No other action substrate allows this (continuous/diffusion have no ordered tokens). NB our k2≈k8 redundancy means useful steering concentrates on the first 1–2 tokens → in practice = value-ranked **exhaustive prefix enumeration** (ties to the BoN refinement above).
- **Discrete tokens → stable LLM-style RL** (GRPO), whereas flow/diffusion VLAs fight RL instability (FPO [2510.09976](https://arxiv.org/pdf/2510.09976), π_RL [2510.25889](https://arxiv.org/html/2510.25889v1)). Discreteness is an advantage the field is trying to recover.
- **Amortized vision** → cheap candidate generation.

**Optional stronger scorer — latent world-model lookahead:** "imagine candidate chunk → predict outcome → score" (AtomVLA [2603.08519](https://arxiv.org/pdf/2603.08519), AHEAD, "Planning in 8 Tokens" [2603.05438](https://arxiv.org/pdf/2603.05438)). More principled than a value-classifier but world-model error compounds.

**Cheap gate — UNIFIED with BoN #7's GATE A.** oracle-BoN = oracle-value: sample N, pick the truly-successful candidate (counterfactual sim) = the "perfect value function". oracle-selection ≫ single → value has headroom → build the Q-function. oracle ≈ single → policy samples not diverse → value has nothing to rank. **One evening, frozen policy; gates #7 AND this at once.**

**Novelty positioning (field is crowded — be narrow & honest):** pure "RL fine-tune VLA" / "value re-rank" is taken (TGRPO [2506.08440](https://arxiv.org/html/2506.08440), V-GPS, VGAS [2602.07399](https://arxiv.org/html/2602.07399v2), AQC). The field scores a **flat whole action**; what NOBODY has is **value over the fidelity/depth axis of an anytime action code** — the unique thing OAT's prefix-decodability exposes. The novelty lives on that axis, not in "value-guided selection".

**Using prefix-decodability (3 mechanisms, honest about which collapses):**
- (M1, weak — DON'T sell) value-guided beam over tokens: branch per token, score decoded prefix, keep top-b. **Undercut by k2≈k8** — A_2≈A_8 in action space → prefix value barely moves after token 2 → beam degenerates. Avoid.
- (M2, strong — value PROBE) decode the cheap prefix `A_2`, score `V(state,A_2)`: high → easy state, execute / long R / don't refine; low → value-critical → refine to `A_8` / replan sooner / best-of-N. A **free criticality detector** before paying for full generation. Value-grounded version of adaptive K/R.
- (M3, strongest — value-of-FIDELITY) prefix-decodability gives the family `{A_1,A_2,A_4,A_8}` = SAME intent at rising fidelity → score `V(state,A_k)` across k → learn the **marginal value of tokens**. Impossible for V-GPS/VGAS/Q-chunking (one flat action, no fidelity axis).

**SCIENTIFIC CONTRIBUTIONS (build the paper on C1, not on SR):**
- **C1 — HEADLINE empirical finding: `reconstruction(k) ≠ value(k)`.** Measured: `k2≈k8` in *reconstruction* (err .138→.124, ~flat). Hypothesis: `value(k)` is **NOT** flat — the marginal value of token-fidelity concentrates at *task-critical* (contact) states, invisible to RMS. Worked example: free-space approach → A_2≈A_8, both succeed, value(k) flat → k=2 suffices; grasp → A_2 vs A_8 differ ~0.05/dim, but that 0.05 = grip-vs-miss → value(k=8)≫value(k=2), while RMS can't see 0.05. So **token fidelity matters where task-VALUE is high (contact), NOT where reconstruction is hard (fast-motion, per Step-0)** — a publishable property of anytime action codes that also explains why every prior reconstruction-based adaptive-depth (ours included) failed. Not preempted (FAST/OmniSAT/VQ-VLA = reconstruction; RL works = fixed representation).
- **C2 — METHOD: value-grounded anytime control.** Allocate tokens (K) and replan horizon (R) by *value-of-fidelity* (M2 probe + M3 family), not reconstruction → revives the dead K/R axes on the correct signal.
- **C3 — POSITIONING: first value / process-reward over a prefix-decodable action code** — "process value" for actions (reasoning-LLM analog), only possible on OAT's ordered anytime code; vs the field's flat whole-action value.

**Gate = headline (one experiment does both):** measure **oracle `value(k)` across states** by counterfactual sim — execute `A_k`, k∈{1,2,4,8}, at states of different phases, record success → `success(k | phase)`. value(k) rises & concentrates at contact states → **C1 confirmed** → headline + method motivation → build the value fn. value(k) flat everywhere → still a publishable *negative* finding ("anytime action fidelity is value-irrelevant on LIBERO") → closes adaptive-depth honestly. Both outcomes ⇒ a paper; experiment is cheap (counterfactual sim on a state sample, like oracle-R / oracle-BoN).

**Risks:** offline-RL value training (credit assignment, distribution shift) is the real work — but it's the field's *proven* path (unlike our reconstruction proxies). De-risk: oracle gate first; then verifier-free / V-GPS-style value before full Q-chunking.

**Preempted-by-survey note (avoid reinventing):** patch-H-OAT ≈ CF-VLA ([2604.24622](https://arxiv.org/abs/2604.24622)); adaptive-replan ≈ StreamVLA ([2602.01100](https://arxiv.org/pdf/2602.01100)) / AQC; tokenizer-fork ≈ FASTer ([2512.04952](https://arxiv.org/html/2512.04952v2)) / OmniSAT ([2510.09667](https://arxiv.org/pdf/2510.09667)). Tokenizer-fork is also LOW-VALUE per our cost analysis (cheapens already-cheap AR, doesn't move SR or the vision-dominated cost).

#### Value-Guided OAT — Phase 1 progress (2026-06-06): branching feasible, pilot harness written

**Counterfactual branching is FEASIBLE (verified).** `my_scripts/test_state_roundtrip.py` PASSED on docker: LIBERO `ControlEnv` exposes `get_sim_state()` / `set_state()` / `regenerate_obs_from_state()` (set_state + sim.forward() + re-render). Deterministic replay MATCH (restore + same actions → same traj to ~1e-4). Restore offset ~1.7e-4 (negligible) AND **cancels in the gap** (both k-branches restore the SAME snapshot). Camera obs: `regenerate_obs_from_state` returns RAW robosuite obs → run through `env._extract_obs(raw)` for the policy format. `cur_step`/`done` are Python attrs on `LiberoEnv` → restore manually.
- Branching needs a SEPARATE sequential single-env harness (NOT the async runner). Obs must be To=2-stacked manually (`stack_last_n_obs`) to match what the policy sees.

**Pilot-on-existence harness WRITTEN: `oat/scripts/branch_value_k.py`.** Design = maximize signal (k=1 vs k=8, R=32, contact-focused, M=5) to answer "is there ANY value-gap, and is it at contact?" before scaling.
- Core `estimate_success(s, chunk, R, M)`: restore → exec chunk[:R] open-loop → continue with full k=8 policy to episode end → success; mean over M.
- Per branch state s (visited by k=8 reference rollout): generate ONE 8-token set → decode at k=1 (`Ac`) and k=8 (`A8`) (same tokens, prefixes) → `gap=p8−p1`. Logs phase (`grip_will_change` = gripper toggles within chunk = grasp/release imminent; `eef_vel`; `gripper_open`) + `recon_gap=||norm(A8)−norm(A1)||` over R.
- Built-in analysis: stratified gap (`grip_change` vs not; `slow_eef` vs `fast_eef`) + **killer check** `corr(value_gap, recon_gap)` (≈0/neg → reconstruction doesn't predict value → headline).
- Run: smoke first (`--n_branch 8 --M 2`), then pilot (`--n_branch 80 --M 5 --R 32 --k_coarse 1`). SLOW (M×2 full continuations per state). Main untested risk = obs format into `obs_encoder` (mirrors the runner's path exactly; debug there if it crashes).
- **Power (M=5):** per-state gap is ~noise (SE~0.31) → only STRATIFIED/aggregate effect resolvable (detect contact-vs-free Δ only if ≥~0.13); killer scatter + per-state need confirmatory M≥20. Pilot = existence + calibration only.
- **Reads:** `gap[grip_change]≫gap[no_change]` & `gap[slow_eef]>gap[fast_eef]` → C1 alive, scale up. `corr(gap,recon)≈0` → headline. gap≈0 everywhere even at k1/R32 → strong negative, stop.

**Smoke run (2026-06-06): pipeline works end-to-end; found+fixed 2 design bugs.** `--n_branch 8 --M 2 --R 32` ran clean (branch→restore→continue→analyze→save) — obs pipeline (To-stacking/dtype/ports) CONFIRMED correct (crash was downstream). Two non-statistical issues fixed:
1. **robosuite horizon bug:** LIBERO `horizon=1000, ignore_done=False`; many continuations on one env without `reset()` overflow robosuite's internal `timestep`→`done` sticks→`step` raises "executing action in terminated episode". `set_state` doesn't touch these Python flags. Fix: in `restore()` reset `ctrl.env.timestep=0` + `ctrl.env.done=False`; set `env.env.env.ignore_done=True` at creation.
2. **phase detector broken:** `grip_will_change` fired on ALL 8 states (sign-flip-anywhere catches gripper-channel noise). Fix: SUSTAINED transition `sign(mean(grip[:4]))≠sign(mean(grip[-4:]))` & both |·|>0.5; also log `ncon` (MuJoCo active-contact count) as a physical contact signal.
3. **R=32 over-floors success** (p_full=0.125 ≪ policy 0.58): R=32 open-loop THROUGH grasp states tanks SR for both k → gap masked (floor effect). Large R amplifies *difficulty*, not the k-signal. Fix: default **R=16** (keep p_full off the floor so a k-gap can show); the analysis now prints `p_full range` + phase coverage to watch this.
- gap/corr from smoke = pure noise (n=8/M=2), ignored. NEXT: re-smoke `--n_branch 16 --M 3 --R 16` to confirm both phase strata present & p_full off-floor, then pilot `--n_branch 80 --M 5 --R 16 --k_coarse 1 --n_tasks 5`.

**Experiment extended to 2x2 (k,R) grid + parallelized (`--n_workers`, spawn, distinct env seeds).** `branch_value_k.py` now measures per state `p(k,R)` for k∈{k_coarse,8}×R∈{R_small,R_large} → `value_R=p(R_small)−p(R_large)` (replan-sooner) AND `value_k=p(8)−p(1)` (C1), stratified by phase (grip_change / eef_vel / ncon). Motivated by the smoke hint that R (not k) might be the live axis at contact.

**PILOT RESULT (2026-06-06, n=100, M=5, R∈{8,32}, k 2 vs 8, parallel): NULL — decisive.** Both effects ≈0, NO contact concentration:
| | overall | grip_change | no_grip | slow_eef | fast_eef |
|---|---|---|---|---|---|
| value_R (k=8) | +0.006 | −0.022 | +0.057 | +0.064 | −0.052 |
| value_k (R=32) | +0.030 | +0.043 | +0.006 | +0.020 | +0.040 |
- value_R ≈0, NOT concentrated at contact (grip_change slightly NEGATIVE; the two contact proxies grip_change vs slow_eef DISAGREE; all ≤0.06, within ~1.5σ). **User's "small R wins at contact" hypothesis NOT supported per-chunk.**
- value_k ≈0 everywhere (~0.02–0.045), no concentration. C1 NOT supported. corr(value_k,recon_gap)=−0.143.
- strat-SE ~0.03–0.05 at n=100/M=5 → would resolve a large concentration (Δ≥0.15 ≈3σ); none exists. Solid null for adaptivity.
- **INTERPRETATION (now ORACLE-backed): per-chunk value of BOTH k and R ≈ 0 because closed-loop replanning corrects any single-chunk decision.** The fixed-sweep gains (SR k4=.496→k8=.58; R8=.635→R32=.44) are **COMPOUNDING over ~30 chunks, NOT localizable per-state.** This UNIFIES every negative (K-predictor, convergence-R, oracle k/R) with one mechanism: closed-loop washes out single-chunk choices → no readable per-state heterogeneity → per-observation adaptive depth/horizon CANNOT beat fixed. Confirmed by counterfactual oracle (ceiling), not heuristics.
- Caveats: ran k_coarse=2 (smaller fidelity contrast than 1) — re-run k_coarse=1 + M=10 for paper-grade rigor (won't change verdict). Overall p~0.29 (grasp-heavy selection) but not floored (range [0,1]).
- **VERDICT: per-observation adaptive K/R program CLOSED-NEGATIVE (oracle-backed).** Paper headline shifts from C1 to the **compounding diagnosis**: "apparent fidelity/horizon value is compounding, not per-state; adaptive depth/horizon can't beat fixed — shown by counterfactual oracle." Live pivots (orthogonal — they SELECT among candidates, can EXCEED SR): #7 best-of-N value-selection, #8 Prefix-Guided Visual.

#### oracle-BoN gate (#7) — POSITIVE (2026-06-07): selection has real headroom (first positive in the project)

`branch_value_k.py` gained `--bon_n N` (oracle best-of-N: sample N full rollouts/state → pass@k curve) + `--temperature/--topk` (diversity injection) + `--bon_isolate` (plan-isolated, realizable ceiling) + `--bon_cap` (continuation cap for speed). Parallel via `--n_workers` (spawn). **Note: GPU offscreen EGL render is the per-step bottleneck → ~6 workers optimal, 16 over-subscribes (733s/state vs 78s/state).**

**BoN pass@k result (n=100, N=8, temp=1.0):** `pass@1=0.188 → @2=.251 → @4=.314 → @8=0.370`, **HEADROOM=+0.182** (success ~doubles with perfect selection). Concentrated at CONTACT, both proxies AGREE (unlike the value(k,R) null where they disagreed): slow_eef head **+0.285**, high_contact **+0.290**; fast_eef +0.080.
- Clean CONTRAST with the adaptivity nulls: varying FIDELITY/HORIZON of one plan (value k,R) ≈0 (replan corrects); but SELECTING among diverse PLANS (different grasp modes) helps a lot, esp. at contact (a bad grasp plan fails unrecoverably). Empirically supports "verification > budget-adaptation".
- Structure: if all states had p=.188, pass@8 would be .81; observed .37 → states bimodal: ~63% "doomed" (c=0, already lost), selection helps the ~37% recoverable.
- **CAVEAT — pass@N is OPTIMISTIC (oracle):** credits any rollout that got lucky in the CONTINUATION, which a verifier (picks the PLAN, not the future) can't control. So +0.182 is an UPPER bound. → built `--bon_isolate` to get the REALIZABLE ceiling (N plans, each scored by M continuations to average out continuation luck; held-out: pick best plan by M/2, score on the other M/2 → unbiased). `realizable headroom = heldout − baseline`. Contact-concentration of pass@N hints it's plan-attributable (luck would be phase-uniform), so optimistic — but bon_isolate is the decisive confirm.
- **plan-isolate `--bon_isolate` implemented** (realizable ceiling): N plans, each scored by M continuations; held-out split (pick best by M/2 `p_sel`, score on other M/2 `p_eval`). Reports `baseline_eval` (random plan on eval half — the clean split-baseline, MAIN), `heldout` (selected plan on eval half), `oracle` (max, biased upper bound). **REALIZABLE headroom = heldout − baseline_eval** (both on the same held-out fold → no selection-leak; user's correction, accepted over heldout−mean(p_all) which slightly inflates the baseline via the selected plan's sel-half).

#### plan-isolate gate — DONE (2026-06-07): realizable headroom SMALL (+0.024), leans NEGATIVE — but temp=1 diversity-limited

**LESSON — `--bon_cap` is UNSAFE on libero10 (= LIBERO-LONG).** A tiny isolate run with `--bon_cap 200` gave ALL ZEROS (96 rollouts, 0 successes): LIBERO-LONG success = full multi-stage completion (~whole episode); capping at branch+200 truncates before completion. **Do NOT use bon_cap here (run to episode end, `--bon_cap 0` default).** Without cap, tiny run gave non-zero (base_e 0.375 / held 0.500) → pipeline OK.

**Plan-isolate RESULT (n=80, N=5 plans, M=6 split 3+3, temp=1, no cap):**
| metric | headroom | |
|---|---|---|
| pass@8 (oracle, 1 continuation each) | **+0.18** | optimistic, includes continuation luck |
| biased-oracle (max of 5 plans, M=6) | +0.098 | + max-of-noise |
| **REALIZABLE (heldout − baseline_eval)** | **+0.024** | honest: pick by 3 samples, score on other 3 |

- baseline_eval=0.197, heldout=0.221, oracle=0.302. **The +0.18 collapses to +0.024 when continuation luck is averaged out (M continuations) and selection-bias removed (held-out).** Most of the BoN headroom was continuation luck, NOT plan quality.
- **Contact-concentration ALSO collapsed**: pass@N had slow_eef +0.285 / high_contact +0.290; realizable has slow_eef +0.047 / high_contact +0.028 / grip_change +0.023 → the "selection helps at contact" story was mostly continuation luck too.
- baseline_eval (0.197) < baseline_allM (0.204) confirms the user's split-baseline correction (all-M baseline is inflated by the selected plan's sel-half).
- **CODE AUDITED — no bug.** N plans sampled from same state; M continuations differ only by continuation sampling; held-out split correct. The `bon_n<=0→4` guard did NOT affect this run (`--bon_n 5` given).

**CONFOUND (keeps it from a clean kill): temp=1 → low PLAN diversity.** The 5 plans were sampled at temp=1 (confident policy → similar grasp plans → similar p_i → max≈mean → small realizable headroom). The pass@N +0.18 came from continuation diversity, but *plan* (first-chunk) diversity at temp=1 is small. This is the "raise temperature if diversity low" point.

**⛔ RESUME — decisive disambiguation: re-run isolate at temp=2.0:**
```
MUJOCO_GL=egl uv run python scripts/branch_value_k.py -c my_models/policy_ep-0250_sr-0.596.ckpt \
  -o my_datasets/iso_t20.npz --n_branch 80 --bon_n 5 --M 6 --bon_isolate --temperature 2.0 \
  --n_tasks 6 --n_workers 10   # ~1.5-2h; watch nvidia-smi (10 workers may over-subscribe; drop to 6-8)
```
- realizable headroom GROWS (e.g. +0.024 → +0.10) → diversity was the bottleneck → plans at temp↑ differ in quality → **verifier viable** (build it; deploy temp↑ + verifier) → positive paper.
- stays ~+0.024 even at temp=2.0 → plan quality genuinely uniform → **selection of plans doesn't help** → NEGATIVE → diagnosis paper (3 adaptivity nulls + "BoN headroom is continuation luck, not plan-attributable").
- Alternative decisive test (cheaper, deployment-relevant, accounts for compounding): **verifier-free best-of-N in real eval** — sample N plans/replan, pick by majority-vote/likelihood/low-entropy, measure episode SR vs baseline. +SR → selection works deployed (free!); ≈0 → dead.

**Script state:** `scripts/branch_value_k.py` modes — grid (2x2 k,R), `--bon_n` (pass@k), `--bon_isolate` (realizable, reports baseline_eval/heldout/oracle). Flags: `--n_workers` (spawn, distinct seeds), `--temperature/--topk` (diversity injection), `--bon_cap` (DON'T use on LIBERO-LONG), `--bon_isolate` guard defaults bon_n=4 if unset. All compile-clean & run-validated. Datasets: `bon_t10.npz` (pass@k +0.18), `iso_gate.npz` (realizable +0.024 @ temp1).

#### LEADING POSITIVE PLAN — Branch-DPO (preference-tune OAT on counterfactual labels), 2026-06-08

**Idea (user's, accepted):** don't SELECT a plan at inference (washed/small headroom); instead **shift the policy DISTRIBUTION** — make successful chunk-types more likely, failed ones less. Single decision is washed by replan, but a distribution shift on ALL decisions **compounds** over ~30 replans. This is the *constructive form of our central diagnosis* and the strongest concrete positive proposal so far.

- **Mechanism:** use the branching harness to collect, per branch state, candidate chunks + counterfactual outcomes → preference pairs (A+ better, A− worse) → fine-tune OAT with **DPO/ranking loss on the discrete token sequences** (`logπθ(A+) > logπθ(A−)`, with frozen πref regularization). Sharper than SFT-on-winners (#2 distillation): contrastive negative + πref anti-drift.
- **Why stronger than inference-BoN:** (a) NOT capped at the per-chunk +0.024 — uniform distribution shift compounds over the episode (same compounding that made fixed-k/R gaps large globally); (b) no inference overhead (single forward at deploy); (c) generalizes ("prefer this chunk-type" in all similar states); (d) **OAT-natural** — discrete ordered tokens → LLM-style DPO (vs awkward for continuous/flow).
- **🔴 CRITICAL label fix (else we train on noise):** the user's pseudocode labels by ONE rollout's success/fail — but our iso showed single-rollout success is MOSTLY continuation luck, not chunk quality. → label each candidate by **M-averaged** `p_i` (M continuations, averages out luck), pair only when `p_i+ − p_i−` is **significant** (above noise). This is exactly the `--bon_isolate` machinery. Preferences must be **plan-attributable**, not luck.
- **Same gate as everything:** needs plan-attributable headroom to exist (temp2/iso). iso held-out +0.024 (luck already removed) → small but NONZERO → there IS a real signal to shift toward, and it compounds. temp=2.0 may grow it. If plan-headroom ≈0 even at temp↑ → pairs mostly non-significant → fall back to RL (#1, signal = task-reward not chunk-preference) / replan-aware (#5).
- **Lit / wedge:** APO/HAPO (action-preference for VLA) + CO-RFT (offline RL for action chunking) + WMPO exist → wedge = **counterfactual-sim GROUND-TRUTH labels** (cleaner than interaction/reward-model preferences) + OAT discrete tokens + our diagnosis as motivation.
- **Verdict:** leading positive plan. ORDER: temp2 (running) → if plan-attributable headroom → **Branch-DPO with M-averaged labels + significance-gated pairs** (preferred over runtime-verifier and over SFT-distillation). This unifies #2 (distillation) + user's idea in the strongest form.

#### ⭐ TURNING POINT (2026-06-08): selection headroom is CONCENTRATED at edge states (inverted-U), not dead
> ⚠️ **SUPERSEDED 2026-06-17 — SINGLE-TASK ARTIFACT.** This edge-concentration was measured on `tasks[0]` only (collection bug); it did NOT replicate multi-task. Edge-lead CLOSED. See «🔴 CRITICAL (2026-06-17)» block below.

**temp=2.0 isolate: realizable headroom stayed +0.024** (same as temp1; biased-oracle grew 0.098→0.137 = luck diversity, but held-out flat) → raising temperature does NOT grow the AVERAGE plan-headroom. Looked NEGATIVE for selection/Branch-DPO.

**BUT the inverted-U re-analysis (`scripts/analyze_criticality.py`, free, no sim) flips it.** Binning per-state criticality (`heldout − baseline_eval`) by recoverability (`baseline_eval`) — the +0.024 average is DILUTED; criticality PEAKS at mid-recoverability (edge states), REPLICATED across both temps in the predicted bin:
| recov bin | temp=1 criticality | temp=2 criticality |
|---|---|---|
| [0.0,0.2] (doomed) | +0.018 (n=56) | +0.023 (n=37) |
| **[0.4,0.6] (EDGE)** | **+0.200 ± 0.073 (n=6)** | **+0.133 ± 0.067 (n=10)** |
| [0.8,1.0] (safe) | −0.025 | −0.075 |
- **Fate-deciding edge states EXIST** (~8–12% of branch states, recoverability≈0.5): there the plan choice has LARGE realizable headroom (+0.13–0.20). The overall +0.024 was diluted by ~56 doomed (p≈0.02, crit~0.02) + safe states.
- **Not a binning/noise artifact:** held-out selects by an INDEPENDENT half (p_sel) and scores on p_eval → if plans were equal-quality, heldout≈baseline → crit≈0 even under Bernoulli variance. So +0.20 = genuine plan-quality difference (selection generalizes across halves). Replication in the *predicted* bin on two temps softens multiple-comparison concern.
- **STRATEGIC FLIP:** selection/value is NOT dead — it's CONCENTRATED at rare edge/fate-deciding states. Better story than pure-negative: *"naive per-obs adaptivity fails because MOST states are non-critical (doomed/safe); a rare band of fate-deciding states carries large headroom, target them."*

**LEADING POSITIVE (revised, prefix-decodability-centric):** edge-focused selection —
1. train a **recoverability head** `V(s)` (on our counterfactual labels) → detect edge states (p≈0.5);
2. on edge states: enumerate ~100 **prefix** candidates (k2≈k8 → meaning is in the first ~2 tokens) → select/prefer best (signal +0.20, not the diluted +0.024); on non-edge: greedy (cheap, action doesn't decide);
3. **Branch-DPO with significance-gated pairs auto-focuses on edge states** (where p_i differ) → trains on the +0.20 signal → compounds. Prefix-decodability = cheap candidate generation, exactly where it matters.

**Caveats / to confirm:** per-bin n small (6–10), peak ~2–2.7σ (replication helps). **NEXT robustness check:** `analyze_criticality.py --xkey baseline` on both npz (bin by all-M mean, more independent) — peak should persist. Then: collect more edge-state samples (target mid-p / larger n_branch) for paper-grade; then build the recoverability head + edge-focused prefix-selection / Branch-DPO.

**Robustness + n=160 run (2026-06-09): edge concentration CONFIRMED with the clean axis.** Ran `iso_big.npz` (n=160; overall realizable +0.011 — even smaller, more doomed states: ~72% have p<0.2). Inverted-U:
- `--xkey baseline` (CLEANER — bins by all-M mean, independent of the eval-half heldout): edge bin [0.4,0.6] = **+0.151 ± 0.055 (n=19, ~2.7σ)**; doomed ~0, safe negative. **This replicates the +0.13–0.20 edge peak a 3rd time at the largest n.**
- `--xkey baseline_eval`: edge bin only +0.073 ± 0.062 (~1.2σ) — LOWER, because binning on the subtrahend (baseline_eval is in `crit=heldout−baseline_eval`) biases the mid-bin estimate DOWN. So `--xkey baseline` is the correct/unbiased axis; baseline_eval underestimates.
- **Verdict: edge-concentration is REAL (3× replicated, clean axis ~2.7σ).** Headroom (+0.15) lives at ~12% fate-deciding states (recoverability≈0.5); ~72% doomed + safe states have ~0 → they dilute the +0.011 mean.
- **HOWEVER — two practical walls for the METHOD:**
  1. **Inference-time edge DETECTION** needs a recoverability head `V(s)` predictable from OBS — but obs-conditioning FAILED for the K-predictor (obs-wall). Risk it fails here too. **Branch-DPO SIDESTEPS this** (significance-gated pairs auto-focus on edge states in TRAINING; no inference detector — policy just absorbs the preference). → prefer Branch-DPO over runtime prefix-selection.
  2. **COST:** clean labels need counterfactual sim (~6h / 160 states per the user). Edge pairs are ~12% of states → getting enough for DPO (~thousands) = very expensive sim. Cheap logged labels are continuation-luck-noisy (our finding) → can't substitute. **This is the real bottleneck.** RL/ReST use cheap task-reward labels (no counterfactual) and target base competence (the dominant factor) → may be more cost-effective than Branch-DPO despite Branch-DPO's cleaner motivation.
- **NET:** edge-concentration is a solid SCIENTIFIC finding (paper diagnosis: most states non-critical, rare edge band carries headroom). For the positive METHOD, the choice is Branch-DPO (clean, no inference detector, but EXPENSIVE counterfactual labels) vs RL/ReST (cheap labels, base competence, weak per-chunk gradient). Given cost, **ReST-first → RL** may beat Branch-DPO in practice; Branch-DPO stays the cleanest-motivated but label-expensive option.

#### 🎉🎉 BREAKTHROUGH (2026-06-10): verifier-free BoN BEATS baseline by +0.11 SR — FIRST POSITIVE

**Result (full budget K=8, same n_test):**
| mode | SR | n_exp |
|------|-----|-------|
| baseline (single sample, `--entropy_threshold 0 --use_k_tokens 8`) | **0.581 ± 0.012** (.592/.568/.584) | 3 |
| **verifier-free BoN N=8 `vote`** (mode-seeking KDE consensus) | **0.690 ± 0.008** (.684/.696) | 2 |

- **Δ = +0.109 SR (~9σ), FREE** (no training, no labels, no oracle — just mode-seeking consensus over 8 samples). baseline 0.581 ≈ paper 0.58 → control is correct → gap is real.
- **N-SWEEP (clean inference scaling law, 3 exp each unless noted):**

  | N | 1 (base) | 4 | 8 (n=2) | 16 |
  |---|---|---|---|---|
  | SR | 0.581 ± 0.012 | 0.662 ± 0.025 | 0.690 ± 0.008 | **0.712 ± 0.012** |
  | Δ vs base | — | +0.081 | +0.109 | **+0.131** |

  Monotone, decelerating (per-doubling +0.081→+0.028→+0.022 ≈ `SR ≈ base + a·log N`) — textbook test-time scaling curve (RoboMonkey-style). N=16 still rising (+0.131, not saturated). All `vote` signal. `medoid` ablation + N=32 pending.
- **Selection EXCEEDS the base policy** (adaptivity never could — it's capped at policy SR). Qualitatively stronger result.
- **WHY it works (reconciles with the null, via COMPOUNDING):** per-chunk plan-selection is small (isolate held-out +0.024) BUT applied at EVERY replan (~34/episode) it COMPOUNDS → +0.11. The same compounding that made fixed-K/R gaps large, now working FOR us. The per-state null (oracle value≈0) is NOT contradicted — isolate measured ONE selection w/ single-sample continuation; deployed BoN selects at every step → continuation is also BoN → compounds. So "+0.024 realizable" was the per-CHUNK bound, NOT the deployed/episode bound.
- **Mechanism of `vote`:** mode-seeking = variance reduction — reject the occasional catastrophic outlier sample each replan, execute the robust consensus mode. Avoiding rare fatal errors over 34 steps compounds to big SR.
- **OAT-substrate fit:** vision amortized (encoded once) → best-of-N cheap on the dominant cost axis. The architectural win RoboMonkey lacks.
- **Earlier "verifier-free ≈0" expectation was WRONG** — conflated per-chunk headroom (+0.024) with deployed compounding headroom (+0.11). Lesson: deployed selection compounds; don't bound it by the single-chunk isolate number.

**⛔ NEXT (confirm + characterize the positive):**
1. **Confirm:** +1-2 more bon exp (currently n=2) for tighter mean±std. Cheap (no training).
2. **N-sweep 4/8/16** → inference scaling-law figure (RoboMonkey-style).
3. **`medoid` vs `vote`** ablation.
4. **Multi-suite** — does the +0.11 generalize (now a POSITIVE to generalize, not just the null).
5. (later) learned verifier — may push above +0.11.
- Commands: baseline `--entropy_threshold 0 --use_k_tokens 8`; bon `--bon_free N --bon_signal {vote,medoid} --use_k_tokens 8`. Both K=8 → apples-to-apples (single vs best-of-N).

**Paper impact:** flips from diagnosis-only (~5-6) → **diagnosis + working positive that EXCEEDS base policy** (~7+). Wedges: (a) selection exceeds BC (adaptivity can't); (b) OAT amortized-vision makes BoN cheap; (c) the compounding explanation ties it to our diagnosis; (d) verifier-free (no training). Novelty still characterization-led but now with a concrete SR win.

#### "Mode-decomposed test-time scaling" — sharpen the BoN wedge vs RoboMonkey (2026-06-13, TO TEST iteratively)

**Framing (the differentiator):** RoboMonkey treats the action as a black box → bolts on *Gaussian perturbation* for diversity + a heavy VLM verifier. OAT's **ordered discrete prefix-decodable** code lets us decompose the action into **mode** (first ~1-2 tokens = *which strategy*) vs **refinement** (tail tokens = *how precise*), and spend the sample budget on the **mode axis** specifically — where the decision lives (k2≈k8 → meaning is in the prefix). Continuous/diffusion VLAs can't decompose mode-vs-refinement (no ordered structure) → can only add undifferentiated noise. Turns "BoN on OAT" (sounds like "RoboMonkey on OAT") into "structured/decomposed selection, only possible on an anytime code".

**Three ideas to test iteratively (ranked; START WITH #2):**
1. **Structured prefix-branching — "diversity by construction".** Instead of N *independent* samples (may mode-collapse at temp=1 → why RoboMonkey adds Gaussian noise), deterministically **branch over top-M of the first token** → M *guaranteed-distinct* plan families → `detokenize(eval_keep_k=k)` decodes each prefix to the full 32-step chunk → select. Global mode coverage for free from the discrete tree, vs RoboMonkey's local jitter around one mode.
2. **⭐ Coarse-to-fine / hierarchical BoN (START HERE — different *structure*, not just signal).** Generate + score candidates on **cheap 2-token prefixes** (each prefix-decodes to a full chunk), pick the best mode, **autoregress the tail only for the winner**. RoboMonkey is *flat* (N full samples, score all); ours is coarse(explore modes)→fine(refine winner) — straight out of the anytime code. NB user's concrete variant: BoN on first 2 tokens → finish chosen seq AR to 4 tokens → optionally last 4 tokens in *parallel* (mentee's task, justified by err k4≈k8; needs retrain/parallel head, quality unverified).
3. **Targeted exploration — temperature on the FIRST token only.** High temp on token 1 (explore modes), greedy tail (clean refinement): "explore where the decision is, exploit the routine". Continuous actions have no "first token" → RoboMonkey perturbs uniformly, can't do this.

**RESULT #2 (coarse-to-fine, 2026-06-14, `--bon_prefix_k`):** select on the cheap prefix, AR-refine the winner's tail. n=3 each, N=8, K=8, `vote`:
| mode | SR | Δ vs baseline | frac of flat gain |
|---|---|---|---|
| baseline (single) | 0.581 | — | — |
| **c2f prefix_k=2** | **0.639 ± 0.021** | +0.058 (~4σ) | ~53% |
| flat BoN N=8 | 0.690 ± 0.008 | +0.109 | 100% |
- **Partial positive:** c2f ≫ baseline (~4σ) → prefix-selection really works → decision/diversity is PARTLY in the first tokens. BUT flat ≫ c2f (~3.8σ) → prefix-selection loses ~HALF the gain → tail tokens 3-8 carry selection-relevant variation. **`k2≈k8` in *reconstruction* ≠ "tail irrelevant for *selection*"** (refutes the clean "select on 2 = select on 8" hypothesis).
- **Cost verdict at prefix_k=2: BAD trade** — saves only AR steps (~3×, but AR = cheap axis, vision dominates → negligible wall-clock) while costing −0.05 SR. #2 only pays off if some prefix_k recovers SR≈flat. → NEXT: prefix_k=4 (err k4≈k8 even closer → should retain more); if k4≈flat there's a clean knee, else the gain needs full 8-token candidate diversity.

**RESULT #2b (prefix_k=4, 2026-06-14, n=2): clean monotone KNEE.**
| mode | SR | gain | % of flat gain | AR steps |
|---|---|---|---|---|
| baseline | 0.581 | — | — | 8 |
| c2f prefix_k=2 | 0.639 | +0.058 | 53% | 22 |
| **c2f prefix_k=4** | **0.674 ± 0.002** (.676/.674/.672, n=3) | +0.093 | **86%** | 36 |
| flat BoN N=8 | 0.690 | +0.109 | 100% | 64 |
- **Knee at k=4 (n=3 confirmed):** selecting on the 4-token prefix recovers **86%** of the BoN gain. Gap vs flat = 0.016, now **significant (~2.7σ)** — NOT noise (prefix_k=4 very tight, stderr ~0.001; flat stderr ~0.006). So the knee is real but does NOT reach flat: tokens 3-8 carry a real ~0.016-SR remainder that full selection captures. Selection-relevant diversity is mostly in tokens 1-4 (k2→k4 = 53%→86%). Knee at 4 (not 2) lines up with the trained pow2 budget k=4.
- **Verdict:** #2 is a real, interpretable result ("4 tokens suffice for selection → 86% of the gain cheaper") but it's a **cost-trade on the CHEAP (AR) axis** — ~1.8× fewer AR steps but AR is dominated by vision, so negligible wall-clock, and it costs ~0.015 SR. Good analysis fact (partially supports mode-decomposition), NOT a headline/SR win. The SR/Pareto win remains the BoN×R coupling, not prefix truncation.

**RESULT #3 (first-token-temp, 2026-06-14, `--bon_first_temp 2.0`, n=3): NULL.** Inject high temp on the FIRST token (mode), greedy tail. ft2 = **0.690 ± 0.015** (.674/.692/.704) ≈ flat BoN N=8 = 0.690. Mean UNCHANGED, variance UP (more first-token exploration → wider outcome spread, same mean — `vote` consensus still picks the mode, no better modes appear). → the first token is already diverse enough at temp=1; "explore the mode harder" does not translate to better selection. Global-temp=2 control skipped (moot — no positive to attribute). 

**MODE-DECOMPOSED PROGRAM CLOSED — no SR gain over plain flat BoN:**
| idea | result |
|---|---|
| #2 coarse-to-fine (prefix_k) | cost-trade on cheap (AR) axis, 86% retention, NOT SR |
| #3 first-token-temp | NULL (≈ flat) |
| #1 prefix-branching | NOT run — likely null + preempted by SoTo (ICML 2026); skip |
flat BoN (+0.11) remains the simple positive. Don't pursue #1. Novelty is NOT in mode-decomposition (SoTo owns ordered-token search); it's in the closed-loop diagnosis. → decisive work = GATE A (criticality-gated, disagreement↔edge) + the diagnosis.

**🔴 PREREQUISITE GATE (cheap, no-sim, before building any of the 3 — same discipline as `diag_convergence_div`):** where does *sampling diversity* live — tokens 1-2 or the tail? `k2≈k8` is a *decoder* property (reconstruction); diversity is a *policy* property. Sample N on shared features, decode, measure (a) fraction of action-space variance from positions 1-2 vs 3-8, (b) how many *distinct* first-tokens top-k actually yields at temp=1. Diversity in prefix → mode-decomposed angle alive. Diversity in tail → prefix-branching misses candidates → scheme collapses.
- **Honest caveats:** (i) all 3 optimize the **AR axis = the CHEAP axis** (vision dominates, already amortized once) → wall-clock win is small in absolute; sell as "cheaper/more-scalable BoN" + the *structural* novelty, NOT as the main latency win (that's vision). (ii) Our +0.11 came from `vote` = variance reduction (reject outliers); *increasing* diversity changes the consensus signal's behaviour → diversity and selection-signal must be tuned **jointly**. (iii) None of these raise the SR ceiling beyond what selection already gives — they make selection cheaper/cleaner, not stronger.

#### SESSION 2026-06-14 — mode-decomposed CLOSED, prior-art survey (everything preempted), GATE A built, novelty=diagnosis

**Mode-decomposed BoN — all 3 ideas done, NONE beat flat BoN on SR** (see RESULT #2/#2b/#3 above): #2 coarse-to-fine = cost-trade on cheap AR axis (86% retention, not SR); #3 first-token-temp = NULL (≈flat, +variance); #1 prefix-branching not run (likely null + SoTo). flat BoN (+0.11) is the simple positive. **The mode-decomposed mechanism is preempted by SoTo anyway.**

**PRIOR-ART SURVEY — the mechanism-novelty path is effectively CLOSED (every family we'd build is already published, mostly 2025-2026):**
| our candidate mechanism | preempted by |
|---|---|
| tree-search / coarse-to-fine over ordered tokens | **SoTo** «(1D) Ordered Tokens Enable Efficient Test-Time Search», arXiv **2604.15453**, **ICML 2026**, EPFL+Apple (Gladstone et al. domain=images; BoN+beam+lookahead+prefix-branching). NOT concurrent — prior art. |
| BoN for VLA | RoboMonkey, MG-Select (verifier-free, ICLR 2026) |
| value-guided MCTS on frozen VLA latents | **V-VLAPS** (2601.00969, +5pp on LIBERO, value-MLP on frozen latents) |
| world-model / MCTS over action chunks | WorldPlanner (2511.03077), VLA-Reasoner (2509.22643), Model-Based Search (2508.12211) |
| cross-replan tree reuse | Model Predictive Trees (2411.15651) |
| adaptive chunk length / horizon | AAC (2604.04161), AQC (2605.05544), StreamVLA |
| discrete-diffusion / masked / parallel action head | **MGP** (2512.09101, on LIBERO — = the mentee's "parallel last-4-tokens" task), Discrete Diffusion VLA (2508.20072) |
| tokenizer fork | FAST/FASTer (2512.04952)/OmniSAT |
| latent reasoning / recurrent test-time depth | **RD-VLA** (2602.07845, latent iterative reasoning, adaptive stop by latent-convergence, 0%→90% with 4 iters), MPCoT (2606.06245) |
| test-time RL adaptation | TT-VLA (2601.06748) |
| **energy-based "thinking" action head** | **EB-VLA** «Energy-Based Action Heads Know When They Don't Know», ICRA 2026 **Workshop**, by the EBT authors (Gladstone). energy=confidence/**OOD detector**, test-time energy minimization. |
| belief/consistency vs compounding | DART, Memory-Consistent NN (2310.06171), RTC/BID |

→ **Lesson (N-times confirmed this session): the VLA/action-token space moves in MONTHS; any "clean" borrowed mechanism is already out.** **Durable novelty = our closed-loop DIAGNOSIS** (compounding-not-per-state, reconstruction≠value, edge-concentration, value≠phase/latent), which none of these do. Method = demonstration, cite these as related work.

**EB-VLA is USEFUL AMMUNITION, not a competitor:** its energy detects **OOD / out-of-distribution novelty**; our **edge/criticality is IN-distribution but fate-deciding** → energy-confidence would MISS edge states. This (a) confirms my caveat that unsupervised-EBT energy = likelihood/confidence, NOT task-value (so a value-EBT still needs the gated value signal); (b) **strengthens the diagnosis**: criticality is latent, missed by OOD/confidence AND reconstruction. Wedge to cite: "energy-OOD (EB-VLA) ≠ decision-criticality (in-distribution)". Possible experiment: does EB-VLA-style energy detect our edge states (predict: no).

**Surviving novelty options (ALL gated on whether criticality/value structure is detectable):**
- **A — criticality-gated compute** (spend BoN/depth/RL only at edge): needs a runtime edge detector. **← GATE A built, testing now.**
- **B — discrete-token RL (GRPO) with counterfactual-clean reward**: exceeds BC ceiling; no per-state detection needed; discreteness = RL stability advantage. Crowded (TGRPO/π_RL/SRPO) → wedge = counterfactual-clean advantages + prefix process-reward.
- **C — value-ordered tokenizer / value-guided masked refinement** (operationalize reconstruction≠value at representation level; nobody orders tokens by task-value): gated on value(k) structure.
- **GFlowNet for action tokens** (sample plans ∝ reward → diverse high-VALUE candidates → better BoN; discrete-sequential = OAT-native; relatively open): attacks BoN's diversity bottleneck.
- **epinet** (cheap epistemic-uncertainty head, single forward): a PRINCIPLED upgrade to the GATE-A detector — isolates EPISTEMIC uncertainty (edge=don't-know-which-plan) from ALEATORIC (doomed-mess), may separate edge-from-doomed where raw sample-disagreement can't.
- "Fix-the-signal" wedge pattern (recurring): take a hot architecture, replace its signal with our finding — value-gated RD-VLA depth, value-guided MGP refinement, value-EBT. All gated on the same value/criticality question.

**GATE A — IMPLEMENTED & smoke-validated (criticality-gated existence test):**
- `branch_value_k.py` isolate mode now logs `disagreement` = mean pairwise L2 among the N sampled candidate plans over the executed prefix (normalizer space, BoN-`vote` geometry) — FREE add-on (same N plans isolate already decodes, no extra sim).
- `scripts/gate_disagreement.py` — offline (no-sim): corr(disagreement, criticality=heldout−baseline_eval), disagreement-by-recoverability-bin (edge should be highest if inverted-U), **edge-vs-doomed gap in σ**, and **AUC** (edge vs all / edge vs doomed) — one detectability number.
- **Smoke (n=3) PASSED code-validation:** `disagreement` written & read; corr/bins/AUC compute correctly (AUC=0 at n=3 is the correct calc, not a bug). Numbers are noise — ignore.
- **Hypothesis:** edge states = policy unsure which plan → candidates DISAGREE more → disagreement detects edge WITHOUT obs (dodges obs-wall). **Coward-FAIL risk:** doomed states may also have high disagreement (flailing) → can't separate → edge-vs-doomed AUC catches this.
- **Full run (~12-14h, on docker):** `branch_value_k.py -c my_models/policy_ep-0250_sr-0.596.ckpt -o my_datasets/iso_gateA.npz --n_branch 120 --bon_n 5 --M 6 --bon_isolate --n_tasks 6 --n_workers 6` (NO --bon_cap on LIBERO-LONG) → then `gate_disagreement.py -i my_datasets/iso_gateA.npz`.
- **Read:** AUC(edge vs doomed) >0.7 / edge-bin disagreement ≫ doomed&safe (>~2σ) → **PASS** → criticality detectable from generation-aware signal → build A (criticality-gated compute). AUC≈0.5 / flat → **FAIL** → edge undetectable from disagreement too → if detection wanted, try **epinet** (isolates epistemic); else fall back to **B (RL, no detection)** or pure **characterization** (latent-criticality undetectability = a strong headline negative).

**GATE A RESULT (2026-06-15, n=120, N=5, M=6): FAIL — and ANTI-signal (decisive).**
- `disagreement` mean=2.286 std=1.452 [0.40, 6.35] (real spread). **AUC(edge vs doomed) = 0.316**, AUC(edge vs all) = 0.336 — **BELOW 0.5 → disagreement ANTI-detects edge.** Bins: doomed=**2.434** (highest), **EDGE=1.692 (LOWEST)**, safe=2.068; edge−doomed gap = **−0.742 (~−1.9σ)**. corr(disagreement, criticality) = **−0.044 ≈ 0** (no direct signal either).
- **Mechanism = the predicted "coward FAIL":** disagreement is a **DOOMED/flailing detector**, not edge. Doomed states → policy samples diverse-but-all-bad plans (aleatoric mess → high disagreement). EDGE states → policy passes them looking **confident** (low disagreement) yet the plan choice is fate-deciding.
- **🎯 DEEP FINDING (stronger than the gate): decision-criticality ANTI-correlates with policy uncertainty.** Edge = "looks confident, but the choice is fatal" → criticality is NOT an uncertainty property; it's a **value-geometry** property (large value-gap between plans), decoupled from how much the policy "doubts".
  - **This likely kills epinet AND energy-OOD too:** epinet measures epistemic uncertainty, energy-OOD measures novelty — but edge states look confident & are in-distribution → both give LOW signal at edge → same failure. **ANY uncertainty/confidence signal is structurally wrong** for criticality; detection would need a learned **value/recoverability** estimate (counterfactual), which is obs-conditioned → obs-wall.
- **Caveat:** this run's edge band (recov∈[0.35,0.65], n=15) had criticality −0.022 (NOT the +0.15 from iso_big) → edge-concentration replicated only weakly here (M=6/n=15 noise; phase-wise mild: slow_eef +0.066, high_contact +0.052; realizable +0.029, consistent with prior iso). But the disagreement conclusion holds regardless (direct corr≈0).
- **VERDICT: criticality-gated compute via cheap uncertainty detection is DEAD** (disagreement anti; epinet/energy structurally doomed by criticality⊥uncertainty). → **don't chase epinet.** Two live paths: **(1) characterization** — headline upgraded: *decision-criticality is decoupled from (anti-correlated with) EVERY confidence/uncertainty signal (obs, physics, generation-disagreement, energy-OOD) — a latent value-geometry property*; **(2) SR number** via detection-free methods (BoN-everywhere +0.11 done; RL / BoN-distillation).

**BoN×R Pareto (supporting EVIDENCE for compounding, NOT novelty — explicitly):** 2D grid `--bon_free {1,8,16}` × `--n_action_steps {8,16,24,32}` (have N=1 R-sweep: 0.635/0.577/0.510/0.440; have N=8@R16=0.690). Missing N=8@R{8,24,32}. Hypothesis "BoN buys longer R": BoN@R=24 ≥ single@R=16 (0.577) at lower vision cost = Pareto win; Δ(BoN) GROWING with R = direct compounding confirmation. Run only if the compounding-evidence figure is wanted — it does not generate novelty.

**Verdict / direction:** STOP hunting new mechanisms (all preempted). Novelty = the DIAGNOSIS. The only live mechanism-sliver = **A (criticality-gated)**, decided by GATE A (running). If GATE A PASS → diagnosis + A (gate=our edge finding, the one component nobody else has). If FAIL → characterization paper (latent criticality, undetectable from obs/disagreement/energy-OOD/reconstruction) + RL/distillation for the SR number. Realistic target = ICRA/CoRL main or strong workshop, contingent on multi-dataset generalization (the user's LATER phase) + diagnosis-led framing. Top-ML unlikely without heavy RL.

#### IDEA 1 (2026-06-15) — OAT as task-aware successive-refinement; is reconstruction the WRONG distortion?

**Framing (adjacent-discipline, NOT preempted):** OAT = a **successive-refinement source code** (prefix-decodable); trained on **reconstruction distortion** (MSE). Our `reconstruction≠value` = "wrong distortion measure" in the rate-distortion / **semantic-communication** sense (semantic-RD is in comms/6G, NOT robotics → novel for action tokens). Mechanism = retrain the tokenizer on a **task/value-weighted distortion** → value-ordered tokens. Companion = idea 2 (Value-of-Information adaptive compute; VoI≠uncertainty → addresses GATE-A's criticality⊥uncertainty). These give the paper a THEORY spine (RD/VoI), formalizing the diagnosis.

**Cheap offline gate (`per_timestep_recon_error.py` extended): d_task(k) vs d_rec(k) rate curves** — `d_task_w(k)=Σ_t w_t e_t(k)/Σ_t w_t` for task-weights. Steeper task-curve than d_rec → tokens carry differential task value → idea 1 alive.

**RESULT (n=100k chunks, autoencode GT through frozen tokenizer):**
| weight | k1 | k2 | k4 | k8 | drop | ratio vs d_rec |
|---|---|---|---|---|---|---|
| uniform (d_rec) | 0.124 | 0.083 | 0.060 | 0.047 | 62% | 1.00 |
| gripper (SHARP change) | 0.244 | 0.122 | 0.075 | 0.057 | 77% | **1.23 (steeper)** |
| delta/jerk (motion ctrl) | ~0.165 | — | — | ~0.054 | ~67% | ~1.07 |
- **Directional positive:** gripper-weighted distortion is steeper + 2× higher at k=1 → grasp steps UNDER-served by the reconstruction tokenizer → a task-aware tokenizer has room to reallocate. First non-null architectural signal.
- **IMPORTANT correction:** the tokenizer's reconstruction is NOT flat (d_rec drops 62%!). The old "k2≈k8 flat" was the **min-k dataset = POLICY-PREDICTION error** (dominated by the policy missing the demo), NOT the tokenizer's rate-distortion. OAT's tokenizer DOES use its rate.
- **🔴 CONFOUND (why "alive" is over-optimistic):** gripper-CHANGE is a **discontinuity** → hard to reconstruct at low k **regardless of value** (sharpness, not task-importance). Conflated. PLUS `value(k)≈0` oracle warns per-chunk fidelity (even grasp) may not change SR (replan washes).
- **DISAMBIGUATION ADDED (smooth grasp weights, offline, pending run):** `grip_smooth` (temporally-smoothed change = grasp REGION), `grip_adj` (smoothed minus the sharp peak = NON-discontinuity neighbor steps). Read: grip_smooth & grip_adj ALSO steeper → grasp REGION under-served = TASK-relevance → idea 1 REAL; only sharp `gripper` steep but grip_adj≈d_rec → SHARPNESS artifact → idea 1 dead. Final decisive (if smooth passes): critic `w=||∂V/∂a||²`.
- **VERDICT so far: weak-positive-but-CONFOUNDED, not a clean PASS.** Run the smooth-weight disambiguation before any tokenizer retrain. Even a FAIL is publishable ("task-critical action info already in early tokens / reconstruction-gap is sharpness not value").

**DISAMBIGUATION RESULT (2026-06-15): idea 1 MECHANISM DEAD (clean).** drop-ratio vs d_rec: gripper(SHARP change) **1.23**, grip_smooth(grasp region) 1.11, **grip_adj(neighbor steps, no discontinuity) 1.06**, delta 1.07, jerk 1.08. → ONLY the sharp gripper-toggle step is steeper; the smooth grasp REGION and its non-discontinuity neighbors ≈ d_rec (≈ motion controls). **The steepness was a SHARPNESS artifact (discontinuity hard to reconstruct at low k), NOT task-relevance.** Task-critical grasp region is reconstructed as well as average → a value-ordered tokenizer has NOTHING to reallocate.
- **Clean confirmation of `reconstruction≠value`, the OTHER direction:** value-relevant action info is the EASY (low-distortion) part → reweighting the distortion doesn't change token allocation. The SR(k) steepness (k4=.496→k8=.58) is **COMPOUNDING** (every chunk slightly worse ×34 replans), NOT per-chunk task-value-of-tokens (per-chunk value(k)≈0).
- Critic w=||∂V/∂a||² would almost certainly confirm (all findings point: value-relevant = low-distortion) → NOT worth building to confirm a clean negative.
- **Idea 1 split:** MECHANISM (value-ordered tokenizer) = DEAD. FRAMING (semantic rate-distortion view) = ALIVE & useful — gives the diagnosis a THEORY spine; this experiment is the clean evidence ("task-distortion ≈ reconstruction-distortion in allocation").
- **Idea 2 (VoI) likely inherits the same null:** VoI(token/R) ≈ 0 where value(k,R)≈0 per-state → VoI-adaptive allocation dead per-state too. VoI useful as FRAMING (formalizes "why adaptive computation fails: VoI≈0"), not as a live mechanism.
- **CONVERGENCE (definitive):** both theory ideas (1=task-RD, 2=VoI) = great FRAMING for the diagnosis, but their MECHANISMS are dead for the same reason as everything else (per-state value washed/flat + value-relevant=easy-to-reconstruct). Durable path = **diagnosis (now with RD/VoI formalization as a strength) + verifier-free BoN positive + multi-dataset**. No mechanism-novelty — exhaustively verified from every angle (K/R adaptivity, mode-decomposed BoN, criticality-gated, EBT/epinet/GFlowNet, value-ordered tokenizer, VoI).

#### Ordering-aware token-level GRPO for OAT — framed as a CONFIRMATORY experiment for the diagnosis (2026-06-15)

**Purpose (the key framing): NOT a gain-grab — a constructive PROOF that completes the diagnosis.** The diagnosis is mostly negative (adaptivity / reconstruction / uncertainty all fail). A value-driven RL positive that works WHERE those failed closes the story: *the right signal is VALUE, applied as a distribution SHIFT (compounds), not per-state adaptivity (washed) nor reconstruction/uncertainty (decoupled).* Design RL to TEST the diagnosis (falsifiable), so even a modest SR gain yields a result.

**Method:** GRPO on OAT's ordered FSQ action tokens. Group = the N candidates `predict_action_bon_free` already samples (vision amortized). Advantage `A_i=(R_i−mean R)/std`; no critic (group mean = baseline). Discrete tokens → stable PG (vs flow/diffusion-VLA RL instability — OAT advantage).
- **Reward (the crux):** learned reward model `R(features, tokens)` (V-VLAPS-style MLP on frozen feats, they get +5pp) trained ONCE on counterfactual labels from `branch_value_k.py`; then cheap to score the group. (counterfactual only for training the reward model, not in the RL loop. Episode-success broadcast = cheap but noisy fallback.)
- **OAT-specific wedge — ordering-aware credit:** `loss = −A·Σ_t w_t·logπ(z_t|z_<t) + β·KL(π‖π_ref)`, `w_t` decreasing with position (early tokens = mode/strategy, carry the outcome since k2≈k8). Concentrate credit on early tokens → sample-efficient. Modest novelty (a credit-weighting); its real value = it's a PROBE (see #2 below).
- **Tools:** TEPO (2604.12736, seq-likelihood aggregation stabilizes sparse token credit), GTPO/GRPO-S (2508.04349, token+seq reward shaping) — fresh 2026, not yet on action tokens.
- **Pipeline:** freeze vision-encoder + tokenizer, train AR head only (5M, cheap/stable); reuse `predict_action_bon_free` (group), `branch_value_k.py` (reward labels), `train_policy.py` (update); iterate (ReST/online).

**5 CONFIRMATORY measurements to build in from the start:**
1. **Compounding:** per-chunk |advantage| (small, ≈oracle +0.024) vs episode-SR gain (large) → confirms gains COMPOUND, not per-chunk. Instruments the central thesis directly.
2. **Decision-in-prefix (ordering-ablation = the probe):** train early-weighted vs uniform vs late-weighted credit. early ≥ uniform > late → mode/decision is in early tokens (k2≈k8). This is what makes ordering-credit worth doing (a TEST, not just a tweak).
3. **reconstruction≠value (constructive):** post-RL, SR↑ while reconstruction-MSE flat/↑ → can improve task-value WITHOUT improving reconstruction → confirms decoupling from the positive side.
4. **criticality⊥uncertainty:** map per-state RL gain by recoverability (edge/doomed/safe) & by disagreement → if gains concentrate at edge AND at "confident"(low-disagreement) states → value lives where uncertainty signals were blind. RL succeeds exactly where uncertainty-adaptivity failed.
5. **value>BC:** RL exceeds BC ceiling → headroom real & accessible via value (consistent with BoN selection).
- **Double-edged (good):** if per-chunk advantage is LARGE / ordering-ablation flat / gains not at edge → challenges compounding / prefix / edge claims. Falsifiable → strong if it confirms.

**Assessment:** novelty LOW-MODEST (GRPO-on-VLA done: TGRPO/π_RL/SRPO; ordering-credit = small twist — but reframed as a PROBE it earns its place). **NOT undermined by our nulls** (value(k)≈0 was per-prefix VALUE; GRPO advantage is group-relative "which token-choice is better" — oracle-BoN realizable +0.024 confirms there's signal; compounds like deployed-BoN). Gain potentially the BIGGEST available (RL exceeds BC, single-sample inference). Cost moderate-high (RL loop + reward model). Rating: standalone method ~5; as confirmatory positive atop the diagnosis → paper ~7. Ceiling set by the diagnosis, not this.

**Recommendation/order:** ReST-distillation FIRST (cheap, reuses train_policy.py, bakes the existing +0.11, capped at BC) → if breaking the BC ceiling is needed, ordering-aware GRPO, **designed as the 5-measurement diagnosis-confirmation** (so it can't "fail" — confirmatory measurements are themselves results). Novelty always = the diagnosis; GRPO/ReST = honestly-borrowed positives.

#### FINE-TUNING METHOD DECISION (2026-06-16): GRPO/GTPO REJECTED → AWR → CRAFT-template

**GRPO/GTPO (incl. paper 2508.04349) REJECTED as overkill/ill-fit:** (1) no free verifiable reward (need a reward model); (2) per-state signal weak+noisy — oracle realizable plan-advantage only +0.024, single-rollout reward dominated by continuation luck → group-relative advantage ≈ noise without M-averaging; (3) online → expensive LIBERO-sim loop; (4) mode-collapse risk (kills the diversity our distribution-shift story needs). GTPO's token-level credit is built for LONG reasoning chains → wasted on 8-token chunks. (Only reusable bit: entropy-weighting as a PROBE for "decision in early tokens".)

**AWR (advantage-weighted SFT) = the lean choice:** offline, `loss=−Σ exp(A_i/β)·Σ_t w_t logπ(z_{i,t})`, A from M-averaged clean counterfactual reward, ordering-credit `w_t` as the prefix-probe, freeze vision+tokenizer, reuse `train_policy.py`. Exceeds BC (offline-RL/EM). Validated as a recipe by Recursive Introspection (NeurIPS'24: BoN + RWR=AWR), SRPO/ForesightFlow (AWR-for-VLA). **BUT its counterfactual signal is the WASHED/biased proxy** (value(k,R)≈0) → naive AWR is limited.

**⭐ CRAFT ([2605.04470](https://arxiv.org/abs/2605.04470), May 2026) = best-fitting TEMPLATE + validates the diagnosis.** It formalizes OUR exact gap: dense-but-biased **counterfactual proxy** vs sparse-but-authentic **closed-loop** advantage → bridges via **proxy-residual** policy-gradient decomposition.
- Mechanism: (1) PG = proxy term (forward-model counterfactual adv, group-normalized) + residual term (closed-loop correction); (2) **critical events flagged by proxy↔reality DISAGREEMENT** (not a detector → partially sidesteps our detection-wall); (3) residual = closed-loop rollouts at flagged events (~**5-10% of trajectories**); (4) **hybrid: offline proxy + online residual**; (5) **EMA-teacher (τ≈0.99) + asymmetric KL** = anti-collapse (our diversity concern). Domain: driving (Bench2Drive); VLA experiments in appendix.
- **Maps to us:** proxy = our sim-counterfactual (ground-truth, NOT model-biased like CRAFT's — but washed-biased instead) OR a learned reward model (cheap, CRAFT-faithful); residual = small closed-loop batch grounds the washed proxy in real compounding outcomes → **directly fixes our washing**; EMA-KL = preserve diversity.
- **CRAFT VALIDATES the diagnosis:** an independent 2026 paper identified the SAME proxy↔closed-loop gap → external confirmation that our compounding/washing finding is a real, recognized problem. Cite it; our counterfactual study characterizes the gap in the manipulation / prefix-decodable setting (they: driving).
- **Caveats:** driving domain (port needed); critical-event-via-disagreement maps imperfectly (their disagreement = forward-vs-behavior; ours = proxy-vs-closed-loop; our per-chunk proxy ≈0 everywhere → disagreement structure differs — VERIFY); it's a template, not drop-in.

**FINAL METHOD PATH:** AWR (cheap, offline, on the washed proxy — quick test if proxy-signal helps at all; likely plateaus per CRAFT's bias point) → **CRAFT-template** (proxy reward-model/AWR + residual closed-loop correction at disagreement points + EMA-KL anti-collapse) as the principled fix when the washed proxy plateaus. Both = borrowed positives; novelty = diagnosis (now externally validated by CRAFT). Cheapest shared first step = a **learned reward/value model on counterfactual labels** (needed by AWR, the proxy, and serves the confirmatory measurements).

#### AWR/ReST PIPELINE — IMPLEMENTED & validated end-to-end (2026-06-16)

Cheapest first version: **ReST/AWR with EPISODE-SUCCESS labels** (no counterfactual sim — only the executed candidate gets a reward = its episode outcome; group-relative/counterfactual is the later expensive upgrade). 3 scripts, all smoke-validated on docker:
- **`scripts/collect_awr_dataset.py`** — sequential LIBERO rollouts (reuses `branch_value_k` env helpers); per replan logs `(features=obs_encoder(obs), executed tokens)`; broadcasts episode success to all chunks. `--bon_n 8` → **BoN-distillation** (logs the vote-selected tokens → bakes the +0.11 BoN policy); `--bon_n 0` → base ReST. Stores features (frozen encoder) not raw obs. Smoke: 200 chunks/7 eps, features (N,2,138), tokens (N,8) ✓.
- **`scripts/train_awr.py`** — advantage-weighted SFT of the AR head ONLY (vision+tokenizer frozen; trains on stored features → no vision recompute). `weight=clip(exp((succ−baseline)/beta))`, `loss=weight·Σ_t w_t·CE + beta_kl·KL(π‖π_ref)` (frozen ref copy = anti-collapse). `--ordering {uniform,early,late}` = the per-token credit w_t = **prefix-probe (confirmatory #2)**. Saves via `workspace.save_checkpoint` (syncs ema_model). Smoke: loss 21.8→19.7, weights 0.33/1.0/2.43, kl~0.5 ✓.
- **`scripts/verify_ckpt.py`** — fast NO-sim check (loads, AR-head L1 diff vs base, generate→detokenize). Smoke: diff 7388 "OK changed", tokens(4,8)→action(4,32,7) ✓.
- **Eval:** existing `eval_policy_sim.py -c <awr.ckpt> --entropy_threshold 0 --use_k_tokens 8`.

**Run plan:** collect (`--n_chunks 10000 --bon_n 8`, ~2h) → train_awr (`--beta 0.5 --beta_kl 0.05 --epochs 5`) → eval single-sample (n=3). **Read:** single-sample AWR >0.581 → distillation works; →0.690 → baked most of BoN at single-sample cost; ≈0.581 → washed-proxy insufficient (expected per CRAFT) → escalate to CRAFT-residual. **Ordering ablation** (early/uniform/late) = free confirmatory probe of "decision in early tokens".

**⛔ RESUME (2026-06-16, after reboot) — AWR run, exact commands.** Pipeline 3 scripts all smoke-validated (collect 200ch ✓, train loss 21.8→19.7 ✓, verify diff 7388/load/generate ✓). `collect_awr_dataset.py` now PARALLELIZED (`--n_workers`, spawn). **Parallel path NOT yet smoke-tested.** Expected result: **single-sample AWR ≈ 0.60–0.63 (+0.02..+0.05 over baseline 0.581)** — modest; full 0.690 unlikely in one cheap iteration (single-sample can't replicate BoN's inference-time selection/variance-reduction; episode labels noisy). ≈0.581 plateau = CRAFT's washed-proxy point → pivot to clean counterfactual labels / ReST-iteration / CRAFT-residual (NOT more data — quantity isn't the ceiling).
```
cd oat
# 0) parallel smoke (verify spawn path, ~min):
MUJOCO_GL=egl uv run python scripts/collect_awr_dataset.py -c my_models/policy_ep-0250_sr-0.596.ckpt \
  -o my_datasets/awr_psmoke.npz --n_chunks 100 --n_tasks 2 --bon_n 0 --n_workers 2
# 1) full collect (BoN-distillation, ~1-1.5h; watch nvidia-smi, drop to 4 if OOM — BoN=8x gen):
MUJOCO_GL=egl uv run python scripts/collect_awr_dataset.py -c my_models/policy_ep-0250_sr-0.596.ckpt \
  -o my_datasets/awr_bon.npz --n_chunks 20000 --n_tasks 10 --bon_n 8 --n_workers 6
# 2) train AWR (offline, GPU only, fast):
uv run python scripts/train_awr.py -i my_datasets/awr_bon.npz -c my_models/policy_ep-0250_sr-0.596.ckpt \
  -o my_models/policy_awr.ckpt --beta 0.5 --beta_kl 0.05 --epochs 5 --ordering uniform
# 3) eval single-sample vs baseline 0.581 / BoN 0.690:
MUJOCO_GL=egl uv run scripts/eval_policy_sim.py -c my_models/policy_awr.ckpt -o eval_out/awr -n 3 \
  --entropy_threshold 0 --use_k_tokens 8
# 4) (free probe) ordering ablation: rerun step 2 with --ordering early / late, eval each. early>=uniform>late => decision in early tokens.
```

#### ⛔⛔ SESSION RESUME (2026-06-09) — full state to continue in a new chat

**ONE-LINE STATE:** per-obs adaptive K/R = oracle-NULL (washed by replan); plan-selection headroom is small on AVERAGE (+0.024) but CONCENTRATED at rare "edge" states (recoverability≈0.5: +0.15, replicated 3×) which are NOT detectable from simple features (look like doomed). **UPDATE 2026-06-10: deployed verifier-free BoN N=8 BEATS baseline +0.11 SR (0.581→0.690) — see BREAKTHROUGH block above. The per-chunk null stands; deployed selection COMPOUNDS over replans.** Paper now diagnosis + working positive (~7/10).

**CONFIRMED FINDINGS (oracle-backed):**
- adaptive K (predictor/entropy/agnostic-mix ≈ fixed; fixed k=4 dominates) — NULL.
- adaptive R (convergence < random ≈ fixed at matched mean) — NULL.
- oracle value(k,R) per-state ≈0 (counterfactual, n=100) — NULL. → benefits are COMPOUNDING not per-state; closed-loop washes single decisions.
- plan-selection: oracle pass@N +0.18 → plan-isolate realizable **+0.024** (held-out, luck removed), temp-invariant (temp1=temp2). MOST of the +0.18 was continuation luck.
- **EDGE-CONCENTRATION (the live positive lead):** criticality (heldout−baseline) PEAKS at mid-recoverability — **+0.15–0.20 at p≈0.5** (clean `--xkey baseline` axis), ~0 at doomed(p≈0)/safe(p≈1). Replicated on iso_gate(temp1), iso_t20(temp2), iso_big(n160). ~12% of states. So the +0.024 average was DILUTED by ~72% doomed + safe.
- **`reconstruction(k) ≠ value(k)`** (k2≈k8 reconstruction but SR(k4→k8)=+0.08) AND **decision-value ≠ semantic phase** (criticality FLAT across grip_change/contact/velocity; edge states physically indistinguishable from doomed — only `step` separates safe=early from doomed/edge=late). → decision-criticality is a LATENT value property, invisible to physics/phase. Strong counterintuitive headline.

**WALLS:**
- edge DETECTION from obs is hard (edge≈doomed physically → need rich V(obs)→recoverability, obs-wall risk like the K-predictor). → gated-selection & compute-saving-on-doomed need this; **selection-EVERYWHERE (ungated) and Branch-DPO SIDESTEP detection.**
- counterfactual labels are EXPENSIVE (~6h/160 states) → Branch-DPO / learned-verifier datasets are the cost bottleneck.

**POSITIVE OPTIONS (all bounded by the modest/edge-concentrated headroom):**
1. **verifier-free BoN (JUST IMPLEMENTED, cheapest, NO dataset)** — sample N plans/replan (vision amortized), pick by free signal. **← RUN THIS NEXT.**
2. learned verifier — needs per-candidate dataset (expensive); 1 collection serves verifier + Branch-DPO + recoverability-head.
3. Branch-DPO (significance-gated edge pairs, expensive labels, sidesteps detection, compounds).
4. RL / ReST (cheap labels, base competence — the dominant factor; ReST reuses `train_policy.py`). Fallback if selection is weak.

**JUST IMPLEMENTED (this session):**
- `OATPolicy.predict_action_bon_free` (verifier-free BoN: amortized vision → N candidates → mode-seeking KDE-density `vote` / `medoid` ranking). Dispatched in `predict_action_adaptive` via `bon_free` kwarg.
- `scripts/eval_policy_sim.py` flags `--bon_free N` `--bon_signal {vote,medoid}`.
- `scripts/analyze_criticality.py` (inverted-U test, `--xkey baseline/baseline_eval`).
- `scripts/characterize_edge.py` (where are edge states — found: no physical signature).
- (earlier) `scripts/branch_value_k.py`: modes grid / `--bon_n` / `--bon_isolate`; flags `--n_workers --temperature --topk --bon_cap`(DON'T use on LIBERO-LONG).

**⛔ RUN NEXT (Step 1 — verifier-free BoN, decides cheaply if selection helps deployed):**
```
# baseline single-sample (if not already have a clean 3-exp number)
MUJOCO_GL=egl uv run scripts/eval_policy_sim.py -c my_models/policy_ep-0250_sr-0.596.ckpt -o eval_out/base -n 3
# verifier-free BoN N=8, mode-seeking
MUJOCO_GL=egl uv run scripts/eval_policy_sim.py -c my_models/policy_ep-0250_sr-0.596.ckpt -o eval_out/bonfree8 -n 3 --bon_free 8 --bon_signal vote
```
- SR(bon_free) >> SR(base) → deployed free-selection helps → invest in learned verifier (Step 2). ≈ base → free signal doesn't capture quality (likely, per our nulls) → either learned verifier (expensive dataset) or pivot to RL/ReST.
- NB verifier-free `vote` ranks by CONSENSUS (proxy for quality, not quality); may pick the frequent-but-not-best mode at edge states. Low expectation; it's the cheapest probe.

**Datasets:** `bon_t10.npz`(pass@k +0.18), `iso_gate.npz`(realizable +0.024 @temp1), `iso_t20.npz`(temp2), `iso_big.npz`(n=160).
**Paper:** lead with "WHERE do decisions matter in closed-loop action-token policies?" (counterfactual study). C2=oracle-null (adaptive inference washed). C3=decision-value concentrated at rare fate-deciding states, value≠phase (latent, not contact). Method = demonstration (selection at edge / Branch-DPO / RL). Needs **multi-suite** generalization to be "phenomenon not our policy". Rating ~5-6 (diagnosis only) → ~7 with a working edge-SR-win + multi-suite. Ceiling ~8 (method not novel, sim-only, single policy).

#### 🔴 CRITICAL (2026-06-17): single-task CONTAMINATION in branch harness → multi-task re-validation → EDGE-LEAD CLOSED, diagnosis SHARPENED

**The bug.** `branch_value_k.py` AND `collect_awr_dataset.py` shared an identical collection bug: the inner `while len(rows) < n_branch` exhausted the WHOLE worker budget on `tasks[0]` before the `for task in tasks` loop ever advanced. So EVERY branch/oracle study (oracle value(k,R), BoN pass@N/isolate, edge-concentration, GATE A) was collected on a SINGLE task — `tasks[0]` of libero10 = `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket`. Confirmed via `Counter(d['task'])`: `awr_bon.npz` (20k chunks) = 1 task; old `iso_*.npz` = 1 task.
- **Fix:** `branch_value_k.py` → **round-robin** over tasks (ONE full reference episode per task per rotation; `ti=env_seed` staggers each worker's start so the union covers all tasks even when a worker does <1 rotation). A per-row task quota was REJECTED: one episode yields many rows, so row-capping either overshoots onto tasks[0] or (mid-episode truncation) biases every state to the early/safe phase → wrecks the recoverability distribution. `collect_awr_dataset.py` → per-task chunk budget (`per_task=ceil(n/n_tasks)`, cumulative `target`) — works there because 1 chunk = 1 row and ~28 rows/episode < per_task. Both verified multi-task on smokes (3 tasks balanced).

**SAFE — was always multi-task** (from `eval_policy_sim.py`→`libero_runner.py`, config `n_test=500` over 10 tasks; `-n`=num_exp repeats, NOT n_test): the load-bearing SR numbers — K-axis nulls (predictor trained on 124k multi-task demos; entropy; agnostic-mix; fixed-k dominates), R-axis nulls (fixed-R sweep; convergence<random), **BoN +0.11 deployed** (0.581→0.690→0.712), pow2/fixed-k, mode-decomposed BoN, d_task. NONE contaminated. **The decisive negative+positive backbone STANDS.**

**CONTAMINATED — single-task `tasks[0]`** (all `branch_value_k` oracle/counterfactual studies): oracle value(k,R)≈0, BoN pass@N +0.18 → isolate realizable +0.024, **edge-concentration +0.15 @ recov≈0.5 (replicated 3× — all on tasks[0])**, GATE A disagreement AUC 0.316, value≠phase.

**Multi-task re-validation (`iso_mt.npz`, n=120, 10 tasks balanced ~12 each, N=5, M=6, isolate):**
| metric | single-task (tasks[0]) | **multi-task (10 tasks)** |
|---|---|---|
| edge bin criticality | **+0.15 ± 0.055** [0.4,0.6) n=19 | **−0.053 ± 0.142** [0.4,0.6) n=5 / +0.033 [0.35,0.65] n=10 |
| inverted-U | clean peak @ p≈0.5 | **GONE** — flat: doomed +0.006 / edge +0.033 / safe +0.034 |
| overall realizable headroom | +0.011..+0.024 | **+0.015 ± 0.015** (consistent, small, general) |
| GATE A AUC(edge vs doomed) | 0.316 (anti) | **0.478 ≈ 0.5 (random)** |
| GATE A edge−doomed disagr gap | −0.742 (−1.9σ) | **+0.080 (0.1σ)** |
- **Edge-concentration does NOT generalize** — it was a `tasks[0]` artifact. GATE A's single-task anti-signal also did NOT replicate (now neutral, AUC~0.5). criticality is FLAT across recoverability multi-task.
- **Caveat:** edge bin underpowered multi-task (n=5–10; 69/120 doomed) → "failed to confirm at low power", not "proven absent". But + GATE A AUC~0.5 + flat criticality → weight of evidence is AGAINST usable per-state structure. Powering the edge bin ≈ 5× overnight (~60h; ~4% of states land mid-recov) and can't be targeted (recoverability≠phase) → not worth it.

**REVISED VERDICT — edge-focused method LEAD CLOSED; diagnosis SHARPENED (and now multi-task).**
- DROP edge-targeting (Branch-DPO-on-edge, recoverability-head `V(s)`, value-gated CRAFT-residual) — unsupported multi-task.
- Sharper/cleaner than the single-task story: **per-observation/per-state structure for adaptive computation is essentially ABSENT multi-task** — not in value(k,R), not in an edge band, not detectable from any cheap signal (disagreement AUC~0.5). The ONLY thing that works is **deployed selection (BoN)**, which compounds a small per-chunk headroom (realizable +0.015) over ~34 replans into +0.11 SR. This is the paper's spine and it is multi-task.
- **Positive method leg = AWR / BoN-distillation** (`collect_awr_dataset.py`→`train_awr.py`): bakes deployed BoN into single-sample; needs NO per-state detection (distribution shift, compounds). The correct fine-tuning path; edge-targeting is not.

**AWR collection + in-harness BoN gain (2026-06-17):** `collect_awr_dataset.py` base-probe (collect harness, single-task pre-fix): base 0.399 → BoN N=8 0.542 = **+0.143** (BoN works in-harness, even bigger than the runner's +0.11; absolute lower because that harness ran one hard task — now fixed to multi-task). Re-collect multi-task with the fix, THEN train AWR. Old single-task `awr_bon.npz`/`awr_base_probe.npz` → discard.

**AWR PROGRESS (2026-06-17, evals RUNNING):** multi-task re-collect DONE & validated (new tool `scripts/validate_awr.py` — coverage/shapes/NaN/tokens/SR/weight-preview): `awr_bon.npz` = 20k chunks, **10 tasks balanced (1.2x)**, per-episode SR **0.716** (BoN-distill multi-task → confirms the old 0.542 was the single-task[0] artifact; per-task SR 0.32–0.93), chunk-weighted baseline 0.569, tokens valid (0–999, 19878/20k uniq), 0 non-finite. Trained AR-head-only (vision+tok frozen): e5/e15/e30 `--beta 0.5 --beta_kl 0.05`. Loss NOT plateaued even @e30 (20.2→13.3) — but that's memorization of 20k pairs, NOT a convergence signal; KL climbs 0.50→0.79 (drift; `beta_kl=0.05` too weak to anchor — contributes 0.04 vs loss 13). **Checkpoint audited VALID** (the smaller 486 vs 667MB file scared us): same 518 keys/shapes/dtype, `obs_encoder` & `action_tokenizer` L1=0 (frozen identical to base), AR-head L1=25898 (trained); smaller size = storage-dedup on re-save (views share storage), NOT data loss. NB `verify_ckpt.py` feeds features directly → does NOT test obs_encoder; the L1=0 check does. **Evals RUNNING:** base/e5/e30 single-sample (`--entropy_threshold 0 --use_k_tokens 8`) vs base **0.581**; BoN8 ref **0.690**. READ: AWR>0.581 → distill works (e5-vs-e30 = fit-vs-drift); ≈0.581 → single-sample washout (epochs won't fix); <0.581 → drift (raise `beta_kl` 0.1–0.2). Ordering-ablation (early/late, free probe) after the epoch winner.

**AWR RESULT (2026-06-17) — FIRST DEPLOYABLE POSITIVE: CLEAN (n=3): e5 = 0.649 ± 0.016, e30 = 0.659 ± 0.006 single-sample, vs base 0.581 (+0.068 / +0.078), ~95% of BoN N=8 (0.690) at 1× inference, capturing ~72% of BoN's gain.** e5 ≈ e30 (e30 marginally higher + much tighter std; epochs NOT the lever) → **~0.66 is the distillation washout-floor**: single-sample can't reproduce BoN's inference-time variance-reduction (robust-mode selection over N samples per replan); the residual ~0.03 gap to BoN is exactly that. Use **e30** (slightly higher, tighter). Constructively confirms the diagnosis: selection is the lever + distribution-shift compounds → baking per-chunk BoN-selection into the weights gives +0.078 over the episode. **Two clean positives now:** BoN +0.11 (inference-time, exceeds BC) AND AWR-distill +0.078 (deployable, single forward). **Imitation-distillation leg DONE & positive.** Epochs saturated (e5≈e30) → next levers are NOT epochs: (a) ordering-ablation early/late on e30 (free confirmatory probe, owed); (b) stronger source — re-collect BoN **N=16** (`awr_bon16.npz`) → does the washout-floor rise?; (c) to BREAK the imitation floor toward/past BoN → **value-method** (Q-chunking QC-FQL, or value-BoN = vote→argmax Q), NOT more imitation.

**BoN inference scaling-law (COMPLETE, 2026-06-19):** N=1 0.581 / N=4 0.662 / N=8 0.690 / N=16 0.712 / **N=32 0.717 ± 0.026**. Per-doubling: +0.028 / +0.022 / **+0.005** → **SATURATED ~0.71–0.72** (N=16 ≈ N=32 within noise). Textbook `SR≈base+a·logN` plateau. **Verifier-free vote ceiling = ~0.72, reached at N=16** (N=32 wasted → operating point N=16). **Implication:** vote (consensus) signal exhausted → to exceed 0.72 needs a LEARNED verifier/value (Q-chunking value-BoN, ranks by quality not consensus). AWR-distill 0.659 is ~0.06 below this ceiling (the inference-selection part single-sample can't reproduce).

**AWR N=16-SOURCE RESULT (2026-06-23, `awr_bon16.npz`, per-episode source SR 0.730 vs N=8's 0.716):** epoch sweep (n=3) — 30ep **0.660 ± 0.003** (= N=8 e30 0.659, NO gain), 100ep **0.692 ± 0.033** (jump!, approaches BoN-N16 ceiling 0.712), 150ep **0.664 ± 0.026** (back down). NON-MONOTONIC + noisy (100ep exps 0.716/0.706/**0.654**; 150ep drop argues against a clean epoch-optimum) → **DON'T trust 0.692 without confirmation.** Two reads: (a) real sweet-spot → N=16+100ep = +0.03 over N=8, nearly matching inference-BoN at 1×; (b) noise → floor method-bound ~0.66.

**RESOLVED (2026-06-25, +4 exp → pooled n=7): AWR16@100ep = 0.684 ± 0.027** (0.716/0.706/0.654/0.644/0.704/0.684/0.680). Verdict = read (a), real but modest: **+0.025 over AWR8 e30 (0.659), ~2.4σ** → stronger source + 100ep genuinely helps (NOT regression to 0.66), but **noisier** (std 0.027 vs AWR8's 0.006 — 100ep+strong-source less stable). **Deployable single-forward ≈ inference-BoN N=8** (0.684 ≈ 0.690): baked the full N=8 BoN gain into one forward pass at NO inference cost; **+0.103 over base**. Falls ~0.03 short of the N=16 ceiling (0.712) = the irreproducible inference-time selection part. **Imitation ceiling located ~0.68:** distill lands single-sample ~0.04–0.05 below the source's per-episode BoN SR (N=8 0.716→0.659; N=16 0.730→0.684); epochs/source don't close that gap → **above 0.68 needs value (chunk-Q just failed on cross-state-only signal — see below) or CRAFT-residual.** AWR-distillation leg = DONE, clean positive. Use **AWR16@100ep (0.684)** as the deployable headline.

**PLAN forward (push toward/beyond the BoN ceiling):** (1) clean e5 number (n=3 + base/BoN anchors); (2) **stronger source** — N=16 (`awr_bon16.npz`) done → marginal/noisy (see above); (3) **ReST** OPTIONAL — mainly fixes off-policy dist-shift (AWR data on base-policy states vs deployed AWR-policy states), likely small here (ceiling looks method-bound, not dist-shift-bound); run 1 iter only as a probe (>+3pp ⇒ dist-shift real; ~0 ⇒ confirms method-bound); imitation-bounded either way. (4) **CRAFT = the ceiling-breaker** — closed-loop residual at proxy↔reality disagreement → can EXCEED the imitation source (AWR/ReST cannot). Order: clean number → N=16 source → CRAFT (ReST optional).

#### ⭐ chunk-Q (Q-chunking) — leading positive-method idea (2026-06-17), BEST fit to our case
**Q-chunking** ([2507.07969](https://arxiv.org/abs/2507.07969), offline→online): policy predicts an h-step **chunk**; critic `Q(s, chunk)` scores the WHOLE chunk; **unbiased h-step TD backup** (target = `Σγ^t r + γ^h Q(s_{t+h}, chunk_{t+h:t+2h})`); behavior-constraint keeps it near offline data. Two variants:
- **QC** = sample N chunks from a BC policy → take `argmax_chunk Q`. **This IS value-guided best-of-N.**
- **QC-FQL** = explicit policy maximizing Q with a W₂ BC-constraint → single-forward deployable.

**Why it's the best fit — it unifies BOTH our positives in principled (value) form:**
| Q-chunking | our analog | the upgrade |
|---|---|---|
| QC (argmax Q over N) | verifier-free BoN (by `vote`) | replace `vote` → **learned chunk-Q** → may beat the vote ceiling (0.72); Q is a better selector than consensus |
| QC-FQL (Q-max policy + BC-constraint) | AWR (imitation-weighted SFT) | optimize **Q (value)** not imitate source → **exceeds the imitation ceiling (0.66)** |
| chunk-as-macro-action | OAT chunk (decode 32, exec R=16) | direct match; h = our R |
| flow BC policy f_ξ | **OAT is already our chunk-sampler** | their flow model NOT needed |

Minimal new infra: just a **chunk-Q critic** (MLP on frozen obs-features + decoded continuous chunk), trained by **chunk-level TD on the collected rollout data** (`awr_bon.npz` format already has features + executed chunk + episode reward). QC needs ONLY the critic (OAT = sampler).

**Why it does NOT contradict our oracle null:** the null was per-SINGLE-chunk swap value≈0 (replan corrects). Q-chunking's Q is the **discounted RETURN** (`r+γ^h Q(next)`) → captures **compounding** at the trajectory level → a DIFFERENT quantity than the washed single-swap → can be informative where the per-chunk oracle was null. And it is **value, not a reconstruction/vote proxy** = the central lesson. Granularity (chunk-level) = exactly where our signal lives.

**Caveats (honest):** (1) method is continuous-action (flow BC) — but for QC this is fine: OAT samples chunks (tokens→decode), Q lives on the **decoded continuous chunk**, OAT replaces their flow BC policy. (2) Q may only help at the rare ~12% critical states (most states non-critical → `argmax Q ≈ vote` there) → **value-BoN might be only marginally > vote-BoN**; the upside rides on the Q distinguishing critical states (edge-detection was hard). (V-VLAPS value-MLP gets +5pp on LIBERO — encouraging.) (3) sparse-reward TD (episode-success reward) is the hard part; h-step backup mitigates. (4) h=16 (our R) is the upper end of their recommended 5–10 — watch. (5) no novelty (borrowed) — positive-demonstration; novelty = diagnosis. But the granularity-match (chunk-Q ⟷ compounding) + "value not proxy" is the cleanest diagnosis→method bridge we have.

**Cheapest principled next step (after AWR numbers):** train chunk-Q critic → plug into BoN selection (`vote → argmax Q`) = QC. TEST: **does value-BoN beat vote-BoN (0.72)?** Beats → Q is good → build QC-FQL single-sample policy (the better-than-AWR deployable, exceeds imitation ceiling). ≈vote → Q can't distinguish chunks → confirms "chunk choice rarely matters" = another diagnosis confirmation. **Stronger fit than CRAFT** (less infra — only the critic; unifies both positives; informative either way). Code ref: github.com/ColinQiyangLi/qc.
- **RL family shortlist for our case** (granularity-matched, discrete/chunk-level): chunk-Q (above), **TGRPO** (trajectory-wise GRPO, reuses BoN group, [2506.08440](https://arxiv.org/html/2506.08440v3)), **RLinf-VLA** (ready framework w/ macro-step credit + action chunking + parallel sim, LIBERO-tested, [2510.06710](https://arxiv.org/html/2510.06710v2)), CO-RFT/VLA-OPD (chunked offline-RL bridges from SFT). Empirical caution: a 2026 study finds **PPO > GRPO > DPO** for VLA (GRPO destabilized by non-stationarity, DPO weak under sparse reward + dist-shift — retroactively justifies dropping Branch-DPO).

**chunk-Q (QC) — IMPLEMENTED (2026-06-25), value-BoN ready to run.** Built the QC variant (value-guided best-of-N; OAT = the chunk sampler, only a critic is new). AST-validated; sim runs on docker.
- `oat/oat/model/chunk_q.py` — `ChunkQ(features [B,To,d], chunk_norm [B,R,D]) → success-value logit`. MLP on `flatten(To·d) ⊕ flatten(R·D)`; feat z-score buffers (self-contained, like `TokenCountPredictor`); chunk passed in **normalizer space** (matches `_bon_select` geometry); `score()`=sigmoid; `from_checkpoint`. ~0.16M params.
- `oat/scripts/train_chunk_q.py` — offline, NO sim. Loads `awr_bon.npz`, **detokenizes tokens→continuous chunk** through the frozen tokenizer, normalizes, scores first `horizon`(=n_action_steps=16) steps. Label = **episode success (MC return, γ=1)** → `Q≈P(success|s,chunk)` under the data policy. Split **by episode** (no chunk leakage); BCE with `pos_weight`; checkpoint by best **val AUC**; reports `val_pred_std` (does Q vary enough to rank candidates?).
- `oat/oat/policy/oatpolicy.py` — `set_chunk_q()` (frozen attach) + `_bon_select` gains a **`'value'`** branch: `argmax_N Q(features, cand)` over the executed prefix (features now passed in from `predict_action_bon_free`). vote/medoid unchanged.
- `oat/scripts/eval_policy_sim.py` — `--chunk_q PATH` attaches the critic; `--bon_signal value` (added to Choice) ranks BoN candidates by Q.
- **Run:** train `uv run python scripts/train_chunk_q.py -i my_datasets/awr_bon.npz -c my_models/policy_ep-0250_sr-0.596.ckpt -o my_models/chunk_q.ckpt --epochs 30`; eval `MUJOCO_GL=egl uv run scripts/eval_policy_sim.py -c my_models/policy_ep-0250_sr-0.596.ckpt -o eval_out/bon_value8 -n 3 --bon_free 8 --bon_signal value --chunk_q my_models/chunk_q.ckpt --use_k_tokens 8`. Bar = **vote-BoN N=8 = 0.690** (and N=16 ceiling 0.712).
- **Read:** value-BoN > 0.690 → Q ranks better than consensus → build QC-FQL single-forward (exceeds imitation/AWR ceiling 0.66). ≈ 0.690 → Q can't out-rank vote → "chunk choice given state rarely matters" (diagnosis confirmation). < vote → offline-Q overestimation on off-distribution candidates.
- **Caveats baked in:** (a) `awr_bon.npz` chunks are **vote-selected** (biased subset) vs inference candidates = full sample distribution → mild train/inference shift; if val_auc decent but value-BoN≈vote, re-collect a **base-ReST** set (`--bon_n 0`, raw samples) for a matched distribution. (b) Only ONE chunk per state in data → no within-state contrast; Q learns cross-state P(success|s,a), within-state ranking via generalization (the standard offline limit; true contrast only from the expensive `branch_value_k` counterfactual set). (c) MC (γ=1) first; h-step chunk-TD is the principled upgrade if MC underperforms.

**chunk-Q RESULT (2026-06-25) — value-BoN FAILS (0.476 < base 0.581 < vote 0.690); NOT a bug, diagnosis-confirmation.** Trained on `awr_bon.npz` (val_auc **0.89**, val_pred_std 0.36, clean fit, best@ep4). Sim value-BoN N=8 = **0.476 ± 0.02** (5 exp) — *below base*. Built `scripts/diag_chunk_q.py` (offline, NO sim: mirrors the inference selection on stored features — within-state Q std, spearman(Q,vote), calib gap). **Decisive diagnostic (n=400):**
  - **calib gap = +0.534** (Q(succ=1)=0.772 vs Q(succ=0)=0.238) → Q well-calibrated **cross-state** (= the AUC 0.89). Code/geometry fine, NOT a bug.
  - **within-state Q std = 0.014** (≈3% of the 0.53 cross-state spread) → Q ≈ **V(state)**, near-flat across the 8 candidates at a state.
  - **spearman(Q, vote) = −0.019 ≈ 0** (51% states <0 = random) → Q is *uninformative* within-state (not anti-ranking); the 0.476<base is argmax-over-noise mildly catching OOD outliers.
- **Mechanism (two compounding causes):** (1) label = **episode-success broadcast** to all chunks of an episode = a **state** property, not a chunk property → Q learns `P(success|state)`, nearly ignoring the chunk input. (2) data = **vote-selected** chunks → Q **never sees outliers**, but vote's +0.11 gain IS outlier-rejection (variance reduction) → that within-state signal is *absent from the training distribution* → unlearnable here.
- **Ties to the diagnosis:** vote beats base by **rejecting rare catastrophic samples**, NOT by ranking quality; a regression-Q on broadcast labels can't reproduce it because chunk-quality-given-state is near-constant (the oracle per-chunk-value≈0 null, again). Clean confirmation: *learned chunk-value can't out-rank consensus because the within-state quality signal is near-zero / missing from BoN-distilled data.*
- **ONE cheap principled retry (preempts "wrong-distribution" reviewer):** re-collect **base-ReST** (`collect_awr_dataset.py --bon_n 0`, raw single samples → data CONTAINS outliers + their outcomes) → retrain Q → value-BoN. Combine bon8+bon16 does NOT fix it (both vote-selected, same missing-outlier hole). Likely still flat (per-chunk value≈0) → then value-selection CLOSED; if it gives within-state signal, build QC-FQL. Expensive-but-correct alternative = counterfactual labels (`branch_value_k`), but oracle already says the ceiling is low.
- **chunk-Q's REAL job = value BASELINE, not selector.** Its flatness-in-action (within-state std 0.014) makes `Q(s, executed_chunk) ≈ V(s)` a valid **action-independent baseline** for AWR/PG (AUC 0.89 = good V(s)). Reused in the CRAFT path below.

#### CRAFT direction — staged build toward breaking the imitation ceiling (2026-06-25)
**Honest reframe (what of CRAFT applies):** CRAFT's literal **per-chunk disagreement-residual likely inherits our null** — oracle says per-chunk *reality* ≈ flat (value≈0), so "correct the proxy where it disagrees with reality per-chunk" has ~nothing to target. We take CRAFT's **spirit**, not its letter. The real ceiling-breaker: AWR's loss `weight·CE` with `weight=exp(adv/β)>0` can only push actions **UP** → bounded by imitation (≤ source ≈ 0.68; confirmed AWR16@100ep=0.684). To exceed needs a **signed** gradient (good↑, **bad↓**) from **closed-loop** outcomes + value baseline + KL anti-collapse. chunk-Q becomes the **V(s) baseline** (its real job; flat-in-action ⇒ valid action-independent baseline).
- **Stage 0 — ReST on-policy probe (cheap, reuse, reads dist-shift):** collect with the AWR policy (`collect_awr_dataset.py -c policy_awr16_e100.ckpt --bon_n 8`) → retrain from base → eval. >+3pp over 0.684 = dist-shift real (on-policy iteration matters); ~0 = ceiling is the loss FORM, not dist-shift → motivates Stage 2.
- **Stage 1 — AWR + chunk-Q value baseline (BUILT, low-risk):** `train_awr.py --critic my_models/chunk_q.ckpt` → real AWR (Peng 2019) `adv = succ − V(s)` instead of constant mean SR → lower-variance per-state advantage. Still positive-weight (safe, imitation-regime). Edit: optional `--critic` decodes executed tokens→chunk, `baseline_vec = ChunkQ.score(feats, chunk)`. Run on `awr_bon16.npz`, same hp as the 0.684 run, eval vs 0.684. > 0.684 → learned baseline helps; ≈ 0.684 → ceiling is the loss form (positive-only weight) → Stage 2 is THE lever (strong motivation, not a failure).
- **Stage 2 — on-policy signed-PG (the ceiling-breaker, NOT yet built):** `loss = −adv·logπ(executed_tokens) + β_kl·KL`, `adv = succ − V_chunkQ(s)`, on **on-policy** rollouts (unbiased signed update; off-policy signed-PG needs importance weighting → avoid). adv<0 pushes bad chunks DOWN → can exceed BC. Iterate collect→update (RLOO/REINFORCE-with-baseline). Small new trainer (`train_pg.py`, mirrors `train_awr.py` with signed loss). Build after Stage 0/1 reads.
- Novelty still = the diagnosis; AWR/PG = borrowed positive-demonstrations. EMA-KL/diversity preserved via the existing `beta_kl` ref-KL.

**STAGE 1 RESULT (2026-06-25) — learned baseline HURTS (0.623 < const-baseline 0.684), NEGATIVE but expected.** AWR16@100ep + chunk-Q V(s) baseline (`--critic`, log: V(s)∈[0,1] mean 0.535, adv∈[−0.99,+1.00], weight∈[0.10,5.16]) → SR **0.623** (n=1, below the entire const-baseline range 0.644–0.716 ⇒ ~2.3σ, likely real). **Why:** (1) the sharp per-state baseline over-weights rare "surprising successes" (succeed where V≈0, weight 5.16) and zeroes the bulk of good vote-chunks in safe states (adv≈0) → **dilutes the broad vote-mode-distillation that actually drives 0.684** (variance reduction needs uniform mode-copying, not advantage-picking); (2) those surprising successes are partly **continuation-luck** (our finding) → the baseline amplifies noise. ⇒ **OFFLINE advantage/value reweighting optimizes the wrong thing** (3rd confirmation: per-state advantage ≈0/luck-corrupted; uniform mode-imitation is what works). **0.684 = the offline imitation ceiling.** Use **const-baseline AWR16@100ep (0.684)** as deployable; the critic's job is V(s) for the on-policy Stage 2, not an offline baseline. **Only lever left to exceed 0.684 = Stage 2 (on-policy signed-PG):** signed update pushes bad DOWN (exceeds BC) + on-policy fresh continuations average out the luck that broke Stage 1. NB two solid positives already exist (BoN +0.11 inference, AWR-distill +0.10 deployable) → Stage 2 is upside/the shot past 0.684, NOT required for the paper.

#### Multi-suite expansion — MIKASA-Robo (2026-06-17, investigating)
Goal: a 2nd benchmark so the diagnosis + BoN positive read as "a phenomenon", not "our one LIBERO policy" (the single biggest lever for ICRA per the venue analysis). Assessing feasibility of running OAT + BoN on MIKASA-Robo (local path `MIKASA-Robo`, docs https://mikasarobo.github.io/).
**FINDINGS (2026-06-17): I/O is a near-drop-in match, but it's a MEMORY benchmark → poor fit for a memoryless OAT.** Compatibility: 2× RGB 128×128 (base+hand) = LIBERO layout; 7D proprio (eef pose+gripper); **7D `pd_ee_delta_pose` action, eval in chunk_size=8** (eerily OAT-shaped); 22.5k demos (PPO+motion-planning) in RLDS/LeRobot v3; sim = **ManiSkill 3.0** (not robosuite). Integration: plug OAT into MIKASA's own eval harness (`benchmarking.py`, expects 7D action chunks) rather than porting; LeRobot→Zarr convert; **FULL retrain** of OAT (tokenizer+policy) on MIKASA demos (LIBERO ckpt won't transfer). **DEALBREAKER:** MIKASA is a memory benchmark (90 tasks, 10 memory types, horizons 25–2160; cue must be retained across delay/occlusion) but OAT is **memoryless (To=2 frames)** → ~0 SR on memory-heavy tasks → no headroom to measure BoN/diagnosis. **VERDICT: poor multi-suite choice** for this paper (different AXIS = memory; high cost for a substrate where OAT can't perform). **Prefer: other LIBERO suites (spatial/object/goal — same robosuite, only retrain) or non-memory ManiSkill/MetaWorld.** If MIKASA anyway: only the Short split + bump To (=architecture change), cheap probe first.

#### R-axis REVISITED — prior-art threat (PACE/DEHP/AAC) + plan (2026-06-17)
Prior art claims adaptive execution-horizon (our R-axis) BEATS fixed → challenges our R-negative (already the weaker one: convergence<random single-signal + single-deviation oracle single-task):
- **PACE** ([2606.00537](https://arxiv.org/html/2606.00537)): training-free, replan at low-speed valleys of the predicted chunk's speed profile; **MATCHED-cost** gains (+6–23pp) on RoboTwin/ALOHA/Franka (π0.5), NOT LIBERO. **BUT lacks the random-horizon control we run** (compares only to best-fixed).
- **DEHP** ([2606.11408](https://arxiv.org/html/2606.11408)): PPO obs+chunk-conditioned horizon head, frozen base; +23% avg but **NOT matched-cost** (replans more) on IsaacLab/FurnitureBench → SR-vs-cost trade-off + phase allocation, not matched-cost adaptivity.
- **AAC** ([2604.04161](https://arxiv.org/html/2604.04161v2)): adaptive chunk size via action-entropy.
**Reconciliation:** different signal (kinematic-phase vs our reconstruction-convergence), different benchmark (contact-rich vs LIBERO pick-place), and PACE OMITS the random control → our STRONG "adaptive R dead" is NOT defensible; scope to "on LIBERO/OAT with proper controls, no benefit". **K-axis UNAFFECTED** (these are R/chunk, not token-count) → lead with K.
**Decision:** R is SECONDARY. Cheap signal-probe suffices; reserve the expensive sustained-oracle.
- **DONE (impl):** PACE speed-valley as `adaptive_r='pace'` in `predict_action_variable_r` (‖EE-delta‖_t normalizer-space → smooth(win 3) → first prominent low-speed valley via scipy `find_peaks`, `r_threshold`=prominence; reads PLAN not obs → dodges obs-wall). `scripts/calib_pace_threshold.py` = offline (no-sim) prominence→meanR calibrator.
- **Decisive triplet (matched mean R≈16, LIBERO, n=3):** PACE @ calibrated prominence vs random @ matched mean vs fixed R=16 (=0.577). PACE > random & > fixed → R REVIVES (build on PACE signal; we'd be first to add the random control on LIBERO); PACE ≈ random → scoped R-negative + direct rebuttal of PACE's missing control. **Match random's mean to PACE's IN-SIM mean R** (demo-calibrated thr may drift on rollout states — read mean R from the pace log, retune random).
- **Oracle plan:** grid-oracle MULTI-TASK (`oracle_grid_mt.npz`, single-deviation value(k) & value(R)) for the K-lead ceiling + reconstruction≠value headline (single-task pilot was on the buggy harness). **Sustained-oracle-R** (= upper bound covering PACE-class on LIBERO) RESERVED (expensive ~days), only if a reviewer demands beyond the signal probe.

**PACE TRIPLET RESULT (2026-06-18, n=3, matched mean R≈19–20): PACE signal is ANTI-informative on LIBERO — threat NEUTRALIZED.**
| mode | SR | mean R |
|---|---|---|
| **PACE** @ prom 0.005 | **0.482 ± 0.012** | 19.25 |
| **random** | **0.537 ± 0.026** | 19.95 |
| **fixed R=20** | **0.553 ± 0.026** | 20.0 |
- **PACE < random by −0.055 (~3σ)** AND < fixed by −0.071 at matched mean R. PACE is the WORST of the three. random ≈ fixed (no Jensen room). **3rd reconstruction/kinematic R-signal to lose to random** (after convergence) → kinematic/reconstruction signals are anti-informative, not just useless.
- **Mechanism of failure (from hist):** PACE bimodal ~30%@R=8 / **~33%@R=32** ("no valley → full open-loop R=32"); on LIBERO "no valley = safe long R" is FALSE (R=32 = worst fixed, 0.44) → over-commits long R. random at same mean beats it → it's the CONDITIONING that's bad, not the mean.
- **For the paper — PACE prior-art threat NEUTRALIZED → SUPPORTING evidence:** we ran PACE's OWN signal on LIBERO with the random control PACE OMITTED → it loses. Rebuttal: "PACE's reported gains don't replicate on LIBERO under proper controls; the signal is anti-informative here." R-negative strengthened (every adaptive-R signal < random at matched cost on LIBERO). Scope: LIBERO; PACE's contact-rich gains stand (benchmark-specific, stronger phase structure).
- **✅ Bulletproof DONE — `pace_raw` RESULT (2026-06-23): R-attack-surface CLOSED.** pace_raw (raw `||EE-translation[:3]||`, thr 0.01, mean R 19.7) = **0.502 ± 0.034** < random 0.537 < fixed 0.553 (~1.4σ below random). Both speed definitions (norm pace 0.482, raw pace_raw 0.502) LOSE to random → **PACE-class anti-informative on LIBERO regardless of speed def** → "you implemented PACE wrong" attack CLOSED. **Full R-axis picture (clean):** convergence < random, pace < random, pace_raw < random — EVERY kinematic/reconstruction R-signal loses to random at matched cost on LIBERO.

**GRID-ORACLE MULTI-TASK RESULT (2026-06-18, `oracle_grid_mt.npz`, n=120, 10 tasks, M=5, k 1-vs-8, R 8-vs-32): single-deviation K/R null GENERALIZES (unlike edge).**
| | single-task pilot | **multi-task** |
|---|---|---|
| value_k (k=1 vs 8) | +0.030 | **+0.043** |
| value_R (R=8 vs 32) | +0.006 | **−0.028** |
| corr(value_k, recon_gap) | −0.143 | **+0.079** |
- **value_k small (+0.043)** even at MAX contrast (1 vs 8 tokens), ~1.5σ above 0 — NOT a perfect zero, but small. **value_R ≈ 0** (−0.028, flat across all phases −0.02…−0.04 → single-deviation R washed, no heterogeneity).
- **value_k tracks FAST-MOTION, NOT contact:** fast_eef +0.087 ≫ slow_eef −0.000; low_contact +0.055 > high_contact +0.023; grip_change +0.043 = no_grip +0.043 (NO grasp effect). → **C1 ("value concentrates at contact") DISCONFIRMED multi-task** — token-fidelity matters for fast motion, consistent with Step-0 (recon-hard = fast-motion ≠ task-critical).
- **corr(value, recon) ≈ +0.08 ≈ 0** → reconstruction doesn't predict value → **`reconstruction≠value` headline holds multi-task.**
- **K-lead read (honest):** single-deviation null GENERALIZES (good — unlike edge). value_k is small AND tracks the WRONG signal (motion not task) AND obs-conditioning captures none of it (agnostic-mix=predictor). Full K-negative = oracle (value_k small, motion-tracking, multi-task) + agnostic-mix (obs-conditioning useless, multi-task) = double-closure. Caveat: value_k +0.043 ≠ 0 (~1.5σ) — state honestly as "small & unexploitable", not "zero".

### TODO next

**Status:** K-axis (token count) adaptivity — **NEGATIVE & CLOSED** (predictor/entropy/agnostic-mix ≈ fixed-k; fixed k=4 dominates). R-axis convergence signal — **NEGATIVE & CLOSED** (2026-06-05: convergence 0.553 < random 0.595 at matched mean R≈14; random ≈ fixed → no Jensen room; see "GATE 1 RESULT"). Reconstruction-based adaptivity (min-k, entropy, convergence) all dead → `reconstruction ≠ value`. Step 0 — done. **Current focus (2026-06-17) = AWR/BoN-distillation positive leg + multi-task diagnosis.** Edge-concentration lead CLOSED (single-task artifact, did NOT replicate multi-task — see «🔴 CRITICAL (2026-06-17)»). Per-state adaptive-compute structure absent multi-task; only deployed selection (BoN +0.11, multi-task) works → bake it into single-sample via AWR. Value-Guided/edge-targeting dropped.

1. **GATE 1 — DONE, NEGATIVE (see "GATE 1 RESULT (2026-06-05)" above).** convergence-R < random < fixed at matched mean → convergence signal anti-informative & dead. R-adaptivity prior now LOW (not formally closed — oracle-R unbuilt, but reconstruction-signal pattern + random≈fixed make it unlikely). → pivot to Value-Guided OAT (#9).
2. **(cheap) Add R=4 to the fixed-R sweep** — find the SR(R) peak / where reactivity saturates.
3. **GATE 2 (if 1 passes):** does a generation-aware patch/instability R-signal beat a simple action-entropy (AAC-style) R-signal? Else it reinvents AAC.
4. **Build (if 1–2 pass): Generation-aware Sparse Residual H-OAT + adaptive R** — coarse OAT4 + generation-aware sparse residual patches + patch-activity→R. GATE 3 = maintain SR (semantic risk: patches target fast-motion, not grasp).
5. **Latency reality check:** `measure_latency_adaptive.py` at `batch_size=1` — confirm R (replan/vision-CNN) is the dominant cost and K (AR tokens) is cheap (motivates focusing on R).
6. **For the paper:** multi-suite LIBERO (spatial/object/goal/long); baselines = fixed-k, fixed-R, token-entropy, action-entropy/AAC, agnostic mixtures, learned controller; Pareto SR-vs-(tokens AND replans/latency). The K-axis negative + pow2 + cost-axis analysis is itself a publishable diagnosis.
7. **PIVOT if R fails — Test-time scaling for OAT (best-of-N over ordered tokens).** See "Idea backlog — Test-time scaling" above. First action = **GATE A (oracle best-of-N)**: sample N candidate chunks/replan (vision shared), pick the truly-successful one in sim; oracle-BoN ≫ single → headroom exists → build verifier (free→learned). Can *exceed* the policy SR (unlike adaptivity). Stronger bet than R: positive-result literature (RoboMonkey/MG-Select/RoVer) + unique OAT fit (amortized perception, prefix tree-search).
8. **PIVOT candidate (vision axis) — Prefix-Guided Visual OAT.** See "Idea backlog — Prefix-Guided Visual OAT" above. Attacks the actual bottleneck (vision), generation-aware (dodges obs-wall), RETRAIN-based (a novelty plus, not a con). First action (cheap, frozen, parallelizable with #7) = **`phase × camera masking` premise gate**: does which-camera-matters vary by plan phase (approach vs contact)? Yes → build the (cost-version) plan-gated conditional perception selector; no → premise dead. Caution: crowded field (VLA-Pruner/LightVLA/Compressor-VLA) → sharpen the coarse-prefix-as-plan wedge; decide Cost↓ vs SR↑ up front.
9. **ACTIVE — ⛔⛔ RESUME: see "SESSION 2026-06-14" block** (above SESSION RESUME 2026-06-09). TL;DR: verifier-free BoN +0.11 confirmed (scaling law to +0.131@N16); mode-decomposed (prefix-branch/coarse-to-fine/first-token-temp) ALL closed — none beat flat BoN. **Mechanism-novelty path effectively CLOSED** — every borrowable family (SoTo/V-VLAPS/WorldPlanner/MGP/RD-VLA/EB-VLA/AAC/…) already published 2025-26 → durable novelty = the closed-loop DIAGNOSIS. **GATE A (criticality-gated compute) BUILT** (`branch_value_k.py` logs `disagreement`; `gate_disagreement.py` AUC) & smoke-validated → **RUN full** (`--n_branch 120 --bon_n 5 --M 6 --bon_isolate`, ~12-14h) → AUC(edge vs doomed) decides: PASS→build A (only live mechanism, gate=our edge finding); FAIL→try epinet detector, else RL (B, no detection) / pure characterization. EB-VLA=energy-OOD≠criticality (ammunition, not competitor). Paper = "where do decisions matter" (latent criticality), ~7 with diagnosis + edge-mechanism or working-positive + multi-dataset generalization (LATER phase).
