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
- `oat/oat/model/token_count_predictor.py` — `TokenCountPredictor` MLP (features → min_k)
- `oat/scripts/train_token_count_predictor.py` — trains the predictor from the `.npz`
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

### TODO next

1. ~~**Analyze `collect_min_k_dataset.py` output**~~ — done; min_k distribution at ε=0.10 checked, looks good.
2. ~~**Build token-count predictor**~~ — model + training script written (`oat/oat/model/token_count_predictor.py`, `oat/scripts/train_token_count_predictor.py`). **Next: train on cluster and check `safe_rate`/`under_rate`/`mean_pred_k`.**
3. ~~**Wire predictor into `predict_action_adaptive`**~~ — done: `predict_action_predictor` + `set_token_predictor` + `eval_policy_sim.py --token_predictor` + `detokenize(eval_keep_k=...)`. Compiles; **not yet run in sim.**
4. **Add `--entropy_threshold` flag to `eval_policy_sim.py`** — for threshold sweep experiments.
5. ~~**End-to-end validation**~~ — done for full / entropy-2.75 / predictor-w2.0 (see "End-to-end results" above). Result: learned ≈ entropy in SR, both ~8–10pp below full budget; cost looks method-independent (compounding error).
6. **GATE (reframed — pow2 finding).** fixed k=5/6 are invalid (untrained budgets). Valid fixed points = **{1,2,4,8}**: run fixed k∈{1,2,4} (`--entropy_threshold 0 --use_k_tokens k`; have k=8≈0.58). Then the real test: compare predictor (w2.0 5.31→0.497, w4.0 6.29→0.559) against the best **obs-agnostic mixture** of {1,2,4,8} at the same mean cost. Adaptivity wins only if obs-conditioning beats the mixing rate. If predictor ≈ agnostic mix → no adaptivity value → pivot to tokenizer (uniform-dropout / FASTer) or analysis paper.
7. **Better heuristic baseline:** action-space stopping — `Δ(a_k, a_{k−1})` convergence (or action entropy, AAC-style) instead of token entropy. Cheap, generation-aware; test on the same frontier.
8. **Method (if gate passes):** learned **per-step refine-or-stop** head conditioned on partial generation, trained on task-grounded labels. Start with **offline logged-bandit** `P(success | features, k)` from fixed-k rollouts (reuse `SR(k)` data); DAgger iterations only if needed. Avoid online RL as the starting point.
9. **Map the knee: run predictor `w=4.0`** (~6 tokens) to complete the current MSE-predictor frontier.
10. **Latency reality check:** `measure_latency_adaptive.py` at `batch_size=1` for true per-sample ms (batched runner latency is bound by `max(k_pred)`, not the mean).
11. **For the paper:** multi-suite LIBERO (spatial/object/goal/long), not just libero10; baselines = fixed-k, token-entropy, action-entropy, learned controller; Pareto SR-vs-tokens + latency.
