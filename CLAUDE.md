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

There are no unit tests; correctness is validated via LIBERO simulation rollouts in `env/env_runner/libero_runner.py`.

### Architecture

OAT has a two-stage training pipeline: **tokenizer training** then **policy training**.

**Tokenizer stage** (`workspace/train_oattok.py`):
- VQ-VAE: `RegisterEncoder` → FSQ quantizer (`tokenizer/oat/quantizer.py`) → `SinglePassDecoder`. FSQ default levels `[8,5,5,5]` give a codebook of 1000 entries.
- Trained on reconstruction loss; checkpoint saved for frozen use in stage 2.
- Alternatives with matching config/workspace/policy triplets: FastTok, BinTok, QuestTok (each in `tokenizer/<name>/`, `workspace/train_<name>tok.py`, `config/train_<name>tok.yaml`).

**Policy stage** (`workspace/train_policy.py`):
- `FusedObservationEncoder` (`perception/fused_obs_encoder.py`) merges `RobomimicRgbEncoder` (CNN vision) + `ProjectionStateEncoder` (MLP state) into a shared embedding.
- Fused embedding → transformer backbone → predicts discrete action token indices from the frozen tokenizer's codebook.
- Two policy heads available: **autoregressive** (`model/autoregressive/transformer.py`, with KV-cache for inference) and **diffusion** (`model/diffusion/`, uses `ConditionalUNet1D` or `TransformerForDiffusion`).
- Evaluation checkpointing keeps top-k by LIBERO task success rate.

**Config system**: Hydra-based under `oat/config/`, with task overrides in `config/task/tokenizer/` and `config/task/policy/`. The `scripts/run_workspace.py` entry point instantiates the workspace class via Hydra. Distributed training uses HuggingFace Accelerate. SLURM scripts are in `slurm/`.

---

## Active research: adaptive token budget for OAT

Goal: cut inference cost of `OATPolicy` by predicting *how many* action tokens are actually needed per observation, instead of always generating the full `max_seq_len` (=10).

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
    -o data/libero/libero10_min_k_features.npz
```

Output: `data/libero/libero10_min_k_features.npz` with keys `features`, `errors`, optionally `task_uids`.

### Methodological notes

- "Circularity" concern (using the trained policy to label data for its own helper) is intrinsic to the task — we are predicting a property *of this policy*. This is the same pattern used by CALM, Adaptive Computation Time, early-exit networks, and speculative decoding.
- MSE labels capture "policy's output stabilized" — a proxy for "this input is easy". They may diverge from "policy succeeds at the task" (the simulation-success label). The plan is to validate the learned predictor end-to-end via LIBERO success rate, treating MSE labels as the cheap-to-collect training signal.
- If MSE-trained predictor degrades success rate, fall back to simulation-success labels (≈10× more expensive: run eval with `use_k_tokens=1..10`, take per-episode min-k where success=True).

### `predict_action_adaptive` details

- Defined in `oat/oat/policy/oatpolicy.py:217-273`. Greedy AR generation **without KV-cache** (calls `self.model()` each step, not `self.model.generate()`). Conscious trade-off for simplicity (~3.5 ms/token overhead vs KV-cache version).
- Entropy computed on **full** distribution (before top-k masking), averaged across batch. Early stop: `if step >= 1 and entropy < threshold` (minimum 2 tokens always generated).
- Default `entropy_threshold=2.75`. Entropy range: `[0, log(1000) ≈ 6.91]` nats (codebook size 1000).
- Returns: `{'action', 'action_pred', 'n_tokens', 'entropies'}`.

### Eval pipeline

- `oat/scripts/eval_policy_sim.py` — CLI entry point. Flags: `-c`, `-o`, `-n`, `-d`, `--temperature`, `--topk`, `--use_k_tokens`. **No `--entropy_threshold` flag** — always uses default 2.75 from `predict_action_adaptive` signature. Adding this flag is a known TODO.
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

- `oat/oat/policy/oatpolicy.py` — `predict_action` (line 170, with KV-cache) and `predict_action_adaptive` (line 217, without KV-cache)
- `oat/oat/env_runner/libero_runner.py` — uses `predict_action_adaptive`, tracks `mean_tokens_used`
- `oat/scripts/eval_policy_sim.py` — prints `mean_tokens_used`
- `oat/scripts/collect_min_k_dataset.py` — label generation for token-count predictor
- `oat/my_scripts/measure_latency_adaptive.py` — latency benchmarking with real obs

### TODO next

1. **Analyze `collect_min_k_dataset.py` output** — check min-k distribution, error curves, decide if ε=0.10 is right.
2. **Build token-count predictor** — small MLP/transformer in `oat/oat/model/` that maps features → predicted min-k. Training script in `oat/workspace/` or `oat/scripts/`.
3. **Wire predictor into `predict_action_adaptive`** — as alternative to entropy threshold: predictor says how many tokens to generate, no per-step entropy check needed.
4. **Add `--entropy_threshold` flag to `eval_policy_sim.py`** — for threshold sweep experiments.
5. **End-to-end validation** — run LIBERO eval with predictor-based adaptive generation, compare success rate vs full-budget baseline.
