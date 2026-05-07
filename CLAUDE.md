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
# Option 1: uv (recommended)
uv sync

# Option 2: conda
bash setup/create_env.sh

# Download model weights
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

# Run tests
python test_blt.py
python test_entropy_model.py
python test_config_parser.py
python test_blt_tokenizer.py
```

### Architecture

The core innovation is **entropy-based dynamic patching**: bytes are grouped into variable-length patches based on a learned entropy model, so high-entropy (unpredictable) regions get more compute.

Three transformer components work together:

1. **LocalEncoder** (`model/local_models.py`) — lightweight transformer that processes raw bytes and produces patch representations.
2. **LatentTransformer** (`model/latent_transformer.py`) — large global transformer that reasons over patch embeddings (the main compute).
3. **LocalDecoder** (`model/local_models.py`) — maps patch representations back to byte predictions.

Key supporting modules:
- **`data/patcher.py`** — implements patching modes (entropy, BPE, space, static, byte). The entropy mode is the core BLT contribution.
- **`entropy_model.py`** — small model that predicts next-byte entropy, used to decide patch boundaries.
- **`train.py`** — distributed training via FSDP + optional tensor parallelism, SLURM-aware, resumable.
- **`args.py`** — all training hyperparameters (TrainArgs, DistributedArgs) defined as dataclasses.
- **`config_parser.py`** — YAML + CLI config merging via omegaconf.

Configuration is YAML-based under `bytelatent/configs/`, parsed with omegaconf and merged with CLI overrides.

---

## OAT (Ordered Action Tokenization)

### Setup

```bash
cd oat
# Option 1: uv (recommended)
uv sync

# Option 2: conda/micromamba
micromamba env create -f conda_env.yaml
uv pip install -e .

# Initialize LIBERO submodule
git submodule update --init --recursive

# Download LIBERO datasets (example)
python scripts/convert_libero_dataset.py
```

### Common Commands

```bash
# Train OAT tokenizer
uv run scripts/run_workspace.py --config-name=train_oattok

# Train OAT policy
uv run scripts/run_workspace.py --config-name=train_oatpolicy

# Evaluate policy in simulation
uv run scripts/eval_policy_sim.py --checkpoint path/to/ckpt --output_dir path/to/out

# Compose multi-task dataset
python scripts/compose_libero_multitask_dataset.py
```

### Architecture

OAT has a two-stage training pipeline: **tokenizer training** then **policy training**.

**Tokenizer stage** (`workspace/train_oattok.py`):
- Learns discrete action tokens via vector quantization (VQ-VAE, `vector-quantize-pytorch`).
- The tokenizer maps continuous robot actions → discrete token indices in a learned codebook.
- Alternatives implemented: FastTok, BinTok, QuestTok (each in `tokenizer/<name>/`).

**Policy stage** (`workspace/train_policy.py`):
- Consumes frozen tokenizer + visual/state observations → predicts action token sequences.
- Two policy heads: **autoregressive** (`policy/autoregressive/`) and **diffusion** (`policy/diffusion/`).
- Observation encoding via `perception/fused_obs_encoder.py` (vision + state fusion).

Key supporting modules:
- **`dataset/zarr_dataset.py`** — loads robot demonstration data from Zarr files.
- **`env/env_runner/libero_runner.py`** — runs trained policies in LIBERO simulation for eval.
- **`common/`** — shared utilities: checkpointing, replay buffer, sequence sampling, logging.
- **`gymnasium_util/`** — gym wrappers including async/sync vectorized envs and video recording.

Configuration is Hydra-based under `config/`, with task-specific overrides in `config/task/`. Distributed training uses HuggingFace Accelerate. SLURM job scripts are in `slurm/`.
