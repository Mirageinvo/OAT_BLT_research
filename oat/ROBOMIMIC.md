# RoboMimic OAT — reproduction guide

Fork goal: reproduce **OAT₈** on RoboMimic (Lift / Can / Square), then run **BoN** and **AWR** as on LIBERO. Paper: [OAT arXiv:2602.04215](https://arxiv.org/abs/2602.04215).

**Branch:** `robomimic` (extends `craft_check` with RoboMimic env/runner/configs).

---

## Paper targets (Table VI, OAT₈)

| Task   | Success rate |
|--------|----------------|
| Lift   | 99.2 ± 0.5% |
| Square | 39.2 ± 2.4% |
| Can    | 80.8 ± 2.3% |
| **Avg**| 73.1 ± 0.5% |

**Protocol:** 5 **training** seeds × **50 rollouts** per seed on the **best** policy checkpoint = 250 episodes per task.

**Eval flags:** `--use_k_tokens 8 --entropy_threshold 0`, `MUJOCO_GL=egl`.

**Common mistake:** `--num_exp 5` repeats eval on **one** checkpoint 5 times. For the paper use **`NUM_EXP=1`** (or omit); `n_test=50` is already in `oat/config/task/policy/robomimic/*.yaml`.

---

## Data (mh, 200 demos, 84×84 RGB)

| Artifact | Path | Notes |
|----------|------|--------|
| Lift HDF5 | `data/robomimic/hdf5_datasets/lift_mh_image.hdf5` | Official mh image download |
| Can / Square HDF5 | `*_mh_image.hdf5` | Raw `demo_v15.hdf5` → extract (see below) |
| Zarr (train) | `data/robomimic/{lift,can,square}_N200.zarr` | 200 episodes subsampled from 300 mh demos |

**Validate:**

```bash
cd oat && source .venv/bin/activate
python scripts/validate_robomimic_data.py
```

**Do not use** old `OAT-RoboMimic-Fine-tune/BLT-OAT/data/robomimic/` (ph `image.hdf5`, symlink `lift_N200.zarr` → `image_N200.zarr`).

### Lift

```bash
bash scripts/download_robomimic_datasets.sh lift
bash scripts/prepare_robomimic_lift.sh convert   # → lift_N200.zarr
```

### Can / Square (no pre-built mh-image URL)

```bash
# raw mh demo_v15 + extract to image_v15.hdf5 (uses .venv_extract + robosuite 1.5 for replay)
bash scripts/extract_robomimic_mh_image.sh can square
bash scripts/prepare_robomimic_can.sh convert
bash scripts/prepare_robomimic_square.sh convert
```

Helpers: `scripts/patch_robomimic_env_args.py`, `scripts/setup_extract_venv.sh`.

Converter: `scripts/convert_robomimic_dataset.py` → `oat/env/robomimic/dataset_conversion.py`.

---

## Stage 1 — Tokenizer (per task, frozen for policy)

Config: `task/tokenizer=robomimic/{lift|can|square}`, `train_oattok.yaml`.

- FSQ levels `[8,5,5,5]`, `num_registers=8`, `token_dropout_mode=pow2` → valid k ∈ {1,2,4,8}
- `training.num_demo=200`, `num_epochs=5001`
- **Top-3 checkpoints** by `test_reconst_mse` (not `latest`)

```bash
# Lift
bash scripts/prepare_robomimic_lift.sh tok
TOK_RUN_DIR=output/... bash scripts/prepare_robomimic_lift.sh tok_resume

# Can / Square
bash scripts/prepare_robomimic_can.sh tok
bash scripts/prepare_robomimic_square.sh tok
```

Hydra entry: `accelerate launch scripts/run_workspace.py --config-name=train_oattok ...`

---

## Stage 2 — Policy (paper-default)

Config: `task/policy=robomimic/{lift|can|square}`, `train_oatpolicy.yaml`.

| Setting | Paper / default |
|---------|------------------|
| `horizon` | 32 |
| `n_obs_steps` | 2 |
| `n_action_steps` | 16 |
| `training.num_demo` | 200 |
| `training.rollout_every` | 100 |
| `task.policy.lazy_eval` | **false** (sim SR during train) |
| Optimizer | AdamW, policy_lr **5e-5**, obs_enc_lr **1e-5** |
| Checkpoints | **top-3** by `mean_success_rate` |
| Train / eval inference | **OAT8**: `policy_use_k_tokens=8`, `policy_entropy_threshold=0` in `env_runner` (no entropy early-stop) |
| Tokenizer | `policy.action_tokenizer.checkpoint=...` (**frozen**) |

**Paper-default train (recommended):**

```bash
export TOKENIZER_CKPT=output/.../ep-xxxx_mse-0.00x.ckpt
CUDA_VISIBLE_DEVICES=0 bash scripts/run_policy_robomimic_paper.sh lift "${TOKENIZER_CKPT}"
# can: scripts/cluster_policy_can_paper.sh on cluster (GPU1, n_parallel_envs=2)
```

**Lift all-in-one:** `bash scripts/prepare_robomimic_lift.sh policy` (optional `POLICY_PROFILE=aggressive` on 2×V100 — see `validate_aggressive_profile.sh`).

**5 seeds:** rerun with `training.seed=0..4` (or `SEED=` in cluster scripts).

---

## Evaluation

```bash
MUJOCO_GL=egl bash scripts/eval_robomimic_policy.sh <policy.ckpt> lift
NUM_EXP=1 bash scripts/eval_robomimic_policy.sh <policy.ckpt> lift   # paper: 50 ep once
```

Underlying: `scripts/eval_policy_sim.py` → `RoboMimicRunner` (`oat/env_runner/robomimic_runner.py`).

Env metadata from HDF5: `oat/env/robomimic/env.py` (`create_env_from_metadata`, `postprocess_visual_obs=False`).

---

## BoN (verifier-free test-time scaling)

Same protocol as LIBERO: shared vision, N candidate chunks, `vote` consensus.

```bash
MUJOCO_GL=egl bash scripts/eval_robomimic_bon.sh <policy.ckpt> lift
# BON_N=8 NUM_EXP=1
```

Implemented in `oat/policy/oatpolicy.py` (`predict_action_bon_free`) via `eval_policy_sim.py --bon_free N --bon_signal vote`.

---

## AWR / BoN-distillation

1. **Collect** rollout chunks (+ optional BoN-selected tokens):

```bash
export POLICY_CKPT=output/.../ep-xxxx_sr-0.9xx.ckpt
bash scripts/prepare_robomimic_lift.sh awr_collect
# → my_datasets/awr_lift_bon.npz (features, tokens, episode success)
```

`scripts/collect_awr_dataset.py` supports **robomimic** and **libero** (detects runner from Hydra config).

2. **Train** AR head only (vision + tokenizer frozen):

```bash
bash scripts/prepare_robomimic_lift.sh awr_train
# scripts/train_awr.py — advantage-weighted SFT, optional --ordering early|uniform|late
```

3. **Eval** single-sample (no BoN at inference):

```bash
bash scripts/prepare_robomimic_lift.sh eval_awr
```

Validate dataset: `scripts/validate_awr.py`. Quick ckpt check: `scripts/verify_ckpt.py`.

---

## Cluster (ccmplanner, Docker)

```bash
# Mac → cluster
bash scripts/sync_to_cluster.sh

# On host
ssh -i ~/.ssh/mipt_lab askhabaliev_gs@100.98.148.137
cd ~/mipt_paper/oat
bash scripts/cluster_ccmplanner.sh setup
bash scripts/cluster_ccmplanner.sh status
```

**tmux sessions (example):**

| Session | Role |
|---------|------|
| `tok_square` | Square tokenizer |
| `policy_lift_paper` | Lift policy, paper-default, GPU0 |
| `policy_can_paper` | Can policy, GPU1 |

Artifact map on cluster: `logs/CLUSTER_ARTIFACT_MAP.txt`.

Torch on V100: `scripts/cluster_ensure_v100_torch.sh` (`OAT_USE_UV_RUN=0` in container).

---

## Key files

| Component | Path |
|-----------|------|
| Env | `oat/env/robomimic/env.py`, `factory.py` |
| HDF5→Zarr | `oat/env/robomimic/dataset_conversion.py` |
| Runner | `oat/env_runner/robomimic_runner.py` |
| Vision | `oat/perception/robomimic_vision_encoder.py` |
| Policy | `oat/policy/oatpolicy.py` |
| Tokenizer configs | `oat/config/task/tokenizer/robomimic/*.yaml` |
| Policy configs | `oat/config/task/policy/robomimic/*.yaml` |
| Pipeline scripts | `scripts/prepare_robomimic_{lift,can,square}.sh` |
| SLURM | `slurm/robomimic/*.slurm` |

---

## Related (other benchmarks on this repo)

- **LIBERO:** default `craft_check` path — `CLAUDE.md`, `scripts/collect_min_k_dataset.py`, adaptive K/R, chunk-Q, etc.
- **MetaWorld:** separate port — `METAWORLD.md`, branch `metaworld` (not required for RoboMimic reproduction).
