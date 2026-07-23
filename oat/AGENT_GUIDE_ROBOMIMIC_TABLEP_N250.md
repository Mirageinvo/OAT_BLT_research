# Agent guide — RoboMimic Table P eval (`n_test=250`)

Literal runbook for an agent on branch **`robocasa`** of  
[`Mirageinvo/OAT_BLT_research`](https://github.com/Mirageinvo/OAT_BLT_research/tree/robocasa).

**Goal:** download HF artifacts → place under `oat/` → run **baseline / BoN N=8 vote / AWR** for **Lift, Can, Square** with **`n_test=250`**, no retraining.

**Do not retrain** tokenizer, policy, or AWR. Eval only.

---

## 0. Hard constraints (read first)

| Rule | Detail |
|------|--------|
| Branch | `robocasa` (RoboMimic scripts + `cluster_matched_triplet.sh`) |
| Workdir | `oat/` (repo root has `blt/` + `oat/`; all commands below are from `oat/`) |
| Lift base ckpt | **`ep-0900_sr-0.930.ckpt`** (run A). **Not** ep-1400. |
| Lift AWR | **`awr_s10000_lift.ckpt`** (matched to ep-0900). **Not** `awr_s10000_lift_ep1400.ckpt`. |
| Inference | OAT8: `--use_k_tokens 8 --entropy_threshold 0`, `T=1.0`, `topk=10` |
| BoN | `--bon_free 8 --bon_signal vote` on the **base** policy (no separate BoN ckpt) |
| AWR | single-sample eval of AWR ckpt (BoN-distill N=8 at train time; **no** `--bon_free` at AWR eval) |
| Render | `MUJOCO_GL=egl` |
| Anti-leak | never use selection seeds `1000–1049` for paper/large report eval |

---

## 1. Seed map (anti-leak)

| Role | Seeds | Use for | ❌ Never for |
|------|-------|---------|--------------|
| Train init | Can/Square `42`; Lift retrain `7` | already baked into ckpts | report SR |
| **Selection (TopK during train)** | `test_start_seed=1000`, `n_test=50` → **`1000–1049`** | historical TopK only | any report / Δ / n250 |
| **Paper Table P (done)** | `test_start_seed=10000`, `n_test=50` → **`10000–10049`** | published matched numbers | — |
| **AWR collect** | `--seed 0` | already done; ckpts on HF | report eval |
| **This n250 re-eval** | `test_start_seed=**11000**`, `n_test=250` → **`11000–11249`** | fresh disjoint report pool | overlap with 1000* or reuse-only 10000–10049 |

**Why 11000:** clears selection `1000–1049` and does not pretend to be the same 50-seed paper pool.  
If you instead set `test_start_seed=10000` with `n_test=250`, episodes are `10000–10249` (includes the original 50 + 200 new) — also anti-leak-OK vs selection, but **not** comparable 1:1 to Table P’s 50-seed mean. Prefer **11000** for a clean large-N probe.

**`-n` / `num_exp`:**

| Mode | Flags | Episodes / method | Notes |
|------|-------|-------------------|--------|
| **Recommended large probe** | `N_TEST=250` `N_EXP=1` | 250 | one SR; wall-clock ≈ one Table-P method |
| Paper-style variance | `N_TEST=250` `N_EXP=5` | 1250 | mean±std over 5 repeats; ~5× wall-clock |

Default below = **`N_TEST=250` `N_EXP=1` `TEST_START_SEED=11000`**.

---

## 2. HuggingFace artifacts

### 2.1 Zarr demos

Repo: [`hackhackhack66666/robomimic_zarr`](https://huggingface.co/datasets/hackhackhack66666/robomimic_zarr)

```bash
cd oat
mkdir -p data/robomimic
huggingface-cli download hackhackhack66666/robomimic_zarr \
  --repo-type dataset \
  --local-dir /tmp/robomimic_zarr_dl

# expected layout after download:
#   lift_N200.zarr/  can_N200.zarr/  square_N200.zarr/
rsync -a /tmp/robomimic_zarr_dl/lift_N200.zarr   data/robomimic/
rsync -a /tmp/robomimic_zarr_dl/can_N200.zarr    data/robomimic/
rsync -a /tmp/robomimic_zarr_dl/square_N200.zarr data/robomimic/
```

### 2.2 Tokenizers + base policies + AWR

Repo: [`hackhackhack66666/robomimic-oattok-policy`](https://huggingface.co/hackhackhack66666/robomimic-oattok-policy)

```bash
cd oat
mkdir -p my_models/hf_rm_tablep
huggingface-cli download hackhackhack66666/robomimic-oattok-policy \
  --repo-type model \
  --local-dir my_models/hf_rm_tablep
```

**Canonical paths after download** (use these exact files):

| Role | Path under `my_models/hf_rm_tablep/` |
|------|--------------------------------------|
| Tok Lift | `tokenizers/lift/ep-1970_mse-0.006.ckpt` |
| Tok Can | `tokenizers/can/ep-0520_mse-0.005.ckpt` |
| Tok Square | `tokenizers/square/ep-0690_mse-0.004.ckpt` |
| **Base Lift (USE THIS)** | `policies/lift/ep-0900_sr-0.930.ckpt` |
| Base Can | `policies/can/ep-1700_sr-0.940.ckpt` |
| Base Square | `policies/square/ep-0700_sr-0.420.ckpt` |
| **AWR Lift (for ep-0900)** | `awr/awr_s10000_lift.ckpt` |
| AWR Can | `awr/awr_s10000_can.ckpt` |
| AWR Square | `awr/awr_s10000_square.ckpt` |

**Ignore for this run:** `policies/lift/ep-1400_sr-0.950.ckpt`, `awr/awr_s10000_lift_ep1400.ckpt` (Lift run B / TopK lock — not this agent task).

Optional convenience copies matching cluster layout:

```bash
cd oat
mkdir -p \
  output/20260704/203215_train_oattok_lift_N200/checkpoints \
  output/20260705/210939_train_oattok_can_N200/checkpoints \
  output/20260706/005048_train_oattok_square_N200/checkpoints \
  output/20260706/173343_train_oatpolicy_can_N200/checkpoints \
  output/20260719/144024_train_oatpolicy_lift_N200/checkpoints \
  output/20260720/215024_train_oatpolicy_square_N200/checkpoints \
  my_models

HF=my_models/hf_rm_tablep
cp -L "$HF/tokenizers/lift/ep-1970_mse-0.006.ckpt" \
  output/20260704/203215_train_oattok_lift_N200/checkpoints/
cp -L "$HF/tokenizers/can/ep-0520_mse-0.005.ckpt" \
  output/20260705/210939_train_oattok_can_N200/checkpoints/
cp -L "$HF/tokenizers/square/ep-0690_mse-0.004.ckpt" \
  output/20260706/005048_train_oattok_square_N200/checkpoints/
cp -L "$HF/policies/can/ep-1700_sr-0.940.ckpt" \
  output/20260706/173343_train_oatpolicy_can_N200/checkpoints/
cp -L "$HF/policies/lift/ep-0900_sr-0.930.ckpt" \
  output/20260719/144024_train_oatpolicy_lift_N200/checkpoints/
cp -L "$HF/policies/square/ep-0700_sr-0.420.ckpt" \
  output/20260720/215024_train_oatpolicy_square_N200/checkpoints/
cp -L "$HF/awr/awr_s10000_"{can,lift,square}.ckpt my_models/
```

### 2.3 HDF5 (required for sim env metadata — not on HF)

Eval creates RoboMimic envs from **mh image HDF5** metadata. Zarr alone is not enough.

```bash
cd oat
# Lift (official mh image)
bash scripts/download_robomimic_datasets.sh lift
# Can / Square mh image extract (see ROBOMIMIC.md)
bash scripts/extract_robomimic_mh_image.sh can square   # if not already present
```

Expect:

```
data/robomimic/hdf5_datasets/lift_mh_image.hdf5
data/robomimic/hdf5_datasets/can_mh_image.hdf5
data/robomimic/hdf5_datasets/square_mh_image.hdf5
```

Validate data (optional):

```bash
source .venv/bin/activate   # or .venv_robocasa if that is the RM env on the machine
python scripts/validate_robomimic_data.py
```

---

## 3. Environment / code

```bash
git fetch origin
git checkout robocasa
git pull origin robocasa
cd oat
# cluster Docker example:
#   docker exec -it oat_mipt_robomimic_askhabaliev_gs bash
#   cd /workspace/oat && source .venv/bin/activate
export MUJOCO_GL=egl
export OAT_USE_UV_RUN=0    # if using system/venv python on V100 cluster
```

Key scripts (do **not** invent new entrypoints):

| Script | Role |
|--------|------|
| `scripts/eval_policy_sim.py` | single eval |
| `scripts/cluster_matched_triplet.sh` | baseline → BoN → AWR wrapper + `summary.json` |
| `scripts/cluster_gpu_env.sh` | sourced by matched triplet |

---

## 4. Eval protocol (n_test=250)

### 4.1 One-liner via matched triplet (preferred)

From `oat/`, for each suite. **Lift must override `BASE_CKPT` to ep-0900** (script default may still point at ep-0900 for `SUITE=lift`, but set explicitly).

```bash
cd oat
source .venv/bin/activate
export MUJOCO_GL=egl

N_TEST=250
N_EXP=1
TEST_START_SEED=11000
N_PARALLEL=4          # lower to 2 if GPU RAM tight / sharing

# --- Lift (ep-0900 + matching AWR) ---
SUITE=lift GPU=0 \
  BASE_CKPT=my_models/hf_rm_tablep/policies/lift/ep-0900_sr-0.930.ckpt \
  AWR_CKPT=my_models/hf_rm_tablep/awr/awr_s10000_lift.ckpt \
  SKIP_AWR=0 FORCE_RERUN=1 \
  N_TEST=${N_TEST} N_EXP=${N_EXP} TEST_START_SEED=${TEST_START_SEED} \
  N_PARALLEL=${N_PARALLEL} \
  OUT_ROOT=output/eval/matched_s${TEST_START_SEED}_n${N_TEST}/lift \
  LOG=logs/matched_s${TEST_START_SEED}_n${N_TEST}_lift_gpu0.log \
  bash scripts/cluster_matched_triplet.sh

# --- Can ---
SUITE=can GPU=0 \
  BASE_CKPT=my_models/hf_rm_tablep/policies/can/ep-1700_sr-0.940.ckpt \
  AWR_CKPT=my_models/hf_rm_tablep/awr/awr_s10000_can.ckpt \
  SKIP_AWR=0 FORCE_RERUN=1 \
  N_TEST=${N_TEST} N_EXP=${N_EXP} TEST_START_SEED=${TEST_START_SEED} \
  N_PARALLEL=${N_PARALLEL} \
  OUT_ROOT=output/eval/matched_s${TEST_START_SEED}_n${N_TEST}/can \
  LOG=logs/matched_s${TEST_START_SEED}_n${N_TEST}_can_gpu0.log \
  bash scripts/cluster_matched_triplet.sh

# --- Square ---
SUITE=square GPU=1 \
  BASE_CKPT=my_models/hf_rm_tablep/policies/square/ep-0700_sr-0.420.ckpt \
  AWR_CKPT=my_models/hf_rm_tablep/awr/awr_s10000_square.ckpt \
  SKIP_AWR=0 FORCE_RERUN=1 \
  N_TEST=${N_TEST} N_EXP=${N_EXP} TEST_START_SEED=${TEST_START_SEED} \
  N_PARALLEL=${N_PARALLEL} \
  OUT_ROOT=output/eval/matched_s${TEST_START_SEED}_n${N_TEST}/square \
  LOG=logs/matched_s${TEST_START_SEED}_n${N_TEST}_square_gpu1.log \
  bash scripts/cluster_matched_triplet.sh
```

What the wrapper runs (in order) for each suite:

1. **baseline** — base ckpt, OAT8  
2. **bon** — same base ckpt + `--bon_free 8 --bon_signal vote`  
3. **awr** — AWR ckpt, OAT8 single-sample  
4. writes `OUT_ROOT/summary.json`

### 4.2 Equivalent raw `eval_policy_sim.py` (Lift example)

```bash
cd oat
export MUJOCO_GL=egl
CKPT=my_models/hf_rm_tablep/policies/lift/ep-0900_sr-0.930.ckpt
AWR=my_models/hf_rm_tablep/awr/awr_s10000_lift.ckpt
SEED=11000
NT=250

python scripts/eval_policy_sim.py -c "$CKPT" \
  -o output/eval/matched_s${SEED}_n${NT}/lift/baseline_n1 \
  -n 1 --n_test ${NT} --test_start_seed ${SEED} \
  --n_parallel_envs 4 \
  --use_k_tokens 8 --entropy_threshold 0 \
  --temperature 1.0 --topk 10

python scripts/eval_policy_sim.py -c "$CKPT" \
  -o output/eval/matched_s${SEED}_n${NT}/lift/bon_n8_n1 \
  -n 1 --n_test ${NT} --test_start_seed ${SEED} \
  --n_parallel_envs 4 \
  --use_k_tokens 8 --entropy_threshold 0 \
  --temperature 1.0 --topk 10 \
  --bon_free 8 --bon_signal vote

python scripts/eval_policy_sim.py -c "$AWR" \
  -o output/eval/matched_s${SEED}_n${NT}/lift/awr_n1 \
  -n 1 --n_test ${NT} --test_start_seed ${SEED} \
  --n_parallel_envs 4 \
  --use_k_tokens 8 --entropy_threshold 0 \
  --temperature 1.0 --topk 10
```

### 4.3 Parallelism tip

- 3 suites in parallel (one GPU each, or share carefully): wall-clock ≈ slowest suite (Square).  
- Within a suite, baseline → BoN → AWR are **sequential** in `cluster_matched_triplet.sh`.  
- To cut wall-clock further, run baseline/BoN/AWR as **three jobs** on different GPUs with the raw commands in §4.2 (same seeds).

Rough wall-clock (V100, `N_EXP=1`, `N_TEST=250`): Square ~6 h/method → ~18 h/suite if sequential; Can/Lift faster.

---

## 5. Outputs to collect

Per suite under `output/eval/matched_s11000_n250/<suite>/`:

```
baseline_n1/eval_log.json
bon_n8_n1/eval_log.json
awr_n1/eval_log.json
summary.json          # if using cluster_matched_triplet.sh
```

Report for each suite:

- `mean_success_rate_mean` ± `mean_success_rate_std` (if `N_EXP>1`)
- Δ_BoN = BoN − baseline (percentage points)
- Δ_AWR = AWR − baseline

**Reference Table P (n_test=50, seed 10000, −n 5)** — for orientation only, not the acceptance target of this n250 run:

| Suite | Baseline | BoN N=8 | AWR |
|-------|----------|---------|-----|
| Can | 76.4±3.0% | 80.8±1.1% | 79.2±7.7% |
| Lift A (ep-0900) | 82.0±3.2% | 84.8±4.6% | 81.2±4.1% |
| Square | 36.0±7.5% | 28.0±9.4% | 31.2±5.2% |

---

## 6. Checklist before claiming done

- [ ] Branch `robocasa`, cwd `oat/`
- [ ] Zarrs in `data/robomimic/{lift,can,square}_N200.zarr`
- [ ] HDF5 mh image present for all three tasks
- [ ] Lift base = **ep-0900**; Lift AWR = **awr_s10000_lift.ckpt** (not ep-1400 pair)
- [ ] `test_start_seed=11000`, `n_test=250` (or documented alternative)
- [ ] No eval used `test_start_seed=1000`
- [ ] Baseline + BoN + AWR eval_log.json exist for lift/can/square
- [ ] Did **not** retrain anything

---

## 7. Explicit non-goals

- Do not run MetaWorld / RoboCasa / LIBERO in this task.  
- Do not upload or retrain AWR (`collect_awr_dataset.py` / `train_awr.py`).  
- Do not use exploratory `output/eval/matched/` (seed 1000) or deleted pre-reset Square artifacts.  
- Do not cite filename `sr-0.930` as paper SR — report only `eval_log.json` / `summary.json`.

---

## 8. Pointers in-repo

- Seed / artifact canon: `RESULTS.md` § Table P — Reproduce manifest  
- RoboMimic setup: `ROBOMIMIC.md`  
- Matched wrapper: `scripts/cluster_matched_triplet.sh`  
- Eval CLI: `scripts/eval_policy_sim.py`
