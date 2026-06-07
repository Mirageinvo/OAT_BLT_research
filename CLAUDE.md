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
- **NEXT: run `--bon_isolate` (N=4, M=6, bon_cap 200, n_workers 6, n_branch 80).** heldout−baseline ≫0 (esp. at contact) → build a verifier (positive paper: test-time selection for OAT, prefix-decodable process-value + amortized vision); ≈0 → +0.18 was continuation luck → verifier can't help (folds into the negative). Optional: temp sweep (1.5/2.0) — does pass@N headroom grow with diversity.

### TODO next

**Status:** K-axis (token count) adaptivity — **NEGATIVE & CLOSED** (predictor/entropy/agnostic-mix ≈ fixed-k; fixed k=4 dominates). R-axis convergence signal — **NEGATIVE & CLOSED** (2026-06-05: convergence 0.553 < random 0.595 at matched mean R≈14; random ≈ fixed → no Jensen room; see "GATE 1 RESULT"). Reconstruction-based adaptivity (min-k, entropy, convergence) all dead → `reconstruction ≠ value`. Step 0 — done. **Current focus = PIVOT to Value-Guided OAT** (value not reconstruction); gate = counterfactual-sim oracle harness (oracle value(k|phase) headline + oracle-BoN + oracle-R add-on).

1. **GATE 1 — DONE, NEGATIVE (see "GATE 1 RESULT (2026-06-05)" above).** convergence-R < random < fixed at matched mean → convergence signal anti-informative & dead. R-adaptivity prior now LOW (not formally closed — oracle-R unbuilt, but reconstruction-signal pattern + random≈fixed make it unlikely). → pivot to Value-Guided OAT (#9).
2. **(cheap) Add R=4 to the fixed-R sweep** — find the SR(R) peak / where reactivity saturates.
3. **GATE 2 (if 1 passes):** does a generation-aware patch/instability R-signal beat a simple action-entropy (AAC-style) R-signal? Else it reinvents AAC.
4. **Build (if 1–2 pass): Generation-aware Sparse Residual H-OAT + adaptive R** — coarse OAT4 + generation-aware sparse residual patches + patch-activity→R. GATE 3 = maintain SR (semantic risk: patches target fast-motion, not grasp).
5. **Latency reality check:** `measure_latency_adaptive.py` at `batch_size=1` — confirm R (replan/vision-CNN) is the dominant cost and K (AR tokens) is cheap (motivates focusing on R).
6. **For the paper:** multi-suite LIBERO (spatial/object/goal/long); baselines = fixed-k, fixed-R, token-entropy, action-entropy/AAC, agnostic mixtures, learned controller; Pareto SR-vs-(tokens AND replans/latency). The K-axis negative + pow2 + cost-axis analysis is itself a publishable diagnosis.
7. **PIVOT if R fails — Test-time scaling for OAT (best-of-N over ordered tokens).** See "Idea backlog — Test-time scaling" above. First action = **GATE A (oracle best-of-N)**: sample N candidate chunks/replan (vision shared), pick the truly-successful one in sim; oracle-BoN ≫ single → headroom exists → build verifier (free→learned). Can *exceed* the policy SR (unlike adaptivity). Stronger bet than R: positive-result literature (RoboMonkey/MG-Select/RoVer) + unique OAT fit (amortized perception, prefix tree-search).
8. **PIVOT candidate (vision axis) — Prefix-Guided Visual OAT.** See "Idea backlog — Prefix-Guided Visual OAT" above. Attacks the actual bottleneck (vision), generation-aware (dodges obs-wall), RETRAIN-based (a novelty plus, not a con). First action (cheap, frozen, parallelizable with #7) = **`phase × camera masking` premise gate**: does which-camera-matters vary by plan phase (approach vs contact)? Yes → build the (cost-version) plan-gated conditional perception selector; no → premise dead. Caution: crowded field (VLA-Pruner/LightVLA/Compressor-VLA) → sharpen the coarse-prefix-as-plan wedge; decide Cost↓ vs SR↑ up front.
9. **ACTIVE — Value-Guided OAT (the central reframe).** See "Idea backlog — Value-Guided OAT" + "Phase 1 progress (2026-06-06)". **Survey lesson: our negatives used reconstruction signals; field positives use VALUE.** Headline C1 = `reconstruction(k) ≠ value(k)` (token fidelity matters at contact, invisible to RMS). **Status: branching feasible (verified), pilot harness `scripts/branch_value_k.py` written.** NEXT = run smoke (`--n_branch 8 --M 2`) then pilot (`--n_branch 80 --M 5 --R 32 --k_coarse 1`) → does value-gap concentrate at contact (grip_change/slow_eef) & decouple from recon_gap? Pilot = existence+calibration only (M=5 → stratified effect only, Δ≥~0.13); confirmatory needs M≥20. Then: build the same harness's oracle-BoN (#7 gate) + oracle-R add-on. C2 = value-grounded adaptive K/R; C3 = process-value over prefix-decodable code.
