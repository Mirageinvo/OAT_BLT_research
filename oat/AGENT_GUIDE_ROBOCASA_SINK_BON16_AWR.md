# Agent guide — RoboCasa `turn_off_sink_faucet`: BoN16/32 + AWR (literal-5)

Literal runbook for a mentee on their own GPU. Same protocol as coffee/close; this track is **turn_off_sink_faucet**.

**Canonical protocol:** `ROBOCASA.md` (branch `robocasa`).  
**This guide** = sink-only handoff: **BoN N∈{16,32}** eval + **AWR distill from BoN16 @ 100 epochs**.

Repo: [`Mirageinvo/OAT_BLT_research`](https://github.com/Mirageinvo/OAT_BLT_research) · branch **`robocasa`**.

---

## 0. Hard constraints (read first)

| Rule | RoboCasa sink (THIS job) | MetaWorld / RoboMimic (DO NOT copy) |
|------|--------------------------|--------------------------------------|
| Report layout | **literal 5 seeds** `10000…10004`, each `-n 1 --n_test 50` | often `-n 5` on **one** `test_start_seed=10000` |
| Aggregation | mean over 5 seed SRs; table = **mean ± SEM**; Δ ± SEM_Δ | mean±std over `-n` repeats |
| AWR train | **`--epochs 100`** · collect **`--bon_n 16`** | common default **30 ep / bon_n=8** (or MW sweep) |
| AWR eval | same literal-5 seeds; **single-sample** (no `--bon_free`) | same idea, different seed layout |
| Selection seeds | TopK was **`2000–2049`** only — **never** report on these | MW/RM often `1000–1049` |
| Collect seed | **`0`** only | — |
| Env / venv | **`.venv_robocasa`** (robosuite **1.5** + robocasa v0.2) | shared `.venv` = robosuite 1.4 — **wrong** |
| Action dim | **Da = 12** | MW/RM = 4 or 7 |
| OAT8 flags | `--use_k_tokens 8 --entropy_threshold 0 --temperature 1.0 --topk 10` | same flags, different env |

**BASE_CKPT (locked):**  
`my_models/robocasa_turn_off_sink_faucet_topk_ep0500_sr0.580.ckpt`  
(TopK @ selection seed 2000, SR=0.580, epoch 500. Do **not** retrain / swap.)

**Already done on cluster (do not redo unless asked):** Wave1 **baseline** (+ **BoN8** when present on HF)  
→ pull from HF (below). Your job starts at **BoN16 / BoN32** (+ AWR).

**Wave1 baseline (5 seeds, locked mean):** **0.524**  
(seeds: 0.52 / 0.58 / 0.58 / 0.46 / 0.48). BoN8 may still be finishing on cluster — if `bon_n8_*` / `summary_literal5.json` are missing on HF, wait for a refresh or run BoN16/32 first; **do not start AWR** until `baseline_seed*/eval_log.json` exist locally.

---

## 1. Anti-leak matrix

| Stage | Seeds | Overlap with report? |
|-------|-------|----------------------|
| TopK / selection (done) | `2000–2049` | ❌ never eval for Table P |
| Baseline / BoN8 / BoN16 / BoN32 / AWR **eval** | **`10000 10001 10002 10003 10004`** | shared across methods (OK) |
| AWR **collect** | **`0`** (+ worker offsets) | ❌ |
| AWR **train** | seed `0` in trainer | ❌ |

❌ Collect on 10000 or 2000.  
❌ Eval with `-n 5` on a single start seed.  
❌ Cite TopK@2000 as paper SR.  
❌ Train AWR with `--epochs 30` for paper sink.

---

## 2. What to download from HuggingFace

### 2.1 Repos

| Role | HF repo | Type |
|------|---------|------|
| **Models / tokenizers** | [`hackhackhack66666/rc-model-two-last`](https://huggingface.co/hackhackhack66666/rc-model-two-last) | model |
| **Zarr + Wave1 logs** | [`hackhackhack66666/rc-last-two`](https://huggingface.co/datasets/hackhackhack66666/rc-last-two) | dataset |

Guide (this file): [`AGENT_GUIDE_ROBOCASA_SINK_BON16_AWR.md`](https://github.com/Mirageinvo/OAT_BLT_research/blob/robocasa/oat/AGENT_GUIDE_ROBOCASA_SINK_BON16_AWR.md) on branch `robocasa`.

Videos under `media/` are **optional**. Paper needs `eval_log.json` only.

### 2.2 Layout

**Models (`rc-model-two-last`):**
```
policies/robocasa_turn_off_sink_faucet_topk_ep0500_sr0.580.ckpt   # ~893M
policies/robocasa_turn_off_sink_faucet_topk_lock.txt
policies/robocasa_turn_off_microwave_topk_ep0500_sr0.620.ckpt     # sibling task
tokenizers/sink_ep-3080_mse-0.002.ckpt
tokenizers/microwave_ep-2720_mse-0.002.ckpt
```

**Dataset (`rc-last-two`):**
```
turn_off_sink_faucet_N200.zarr.tar     # ~1.2G — tar -xf → zarr/
turn_off_microwave_N200.zarr.tar
wave1_turn_off_sink_faucet/            # when uploaded (baseline ± bon_n8)
wave1_turn_off_microwave/
```

### 2.3 Mentee: download + place under `oat/`

```bash
cd oat
mkdir -p data/robocasa my_models my_datasets \
  output/eval/matched_s10000/robocasa/turn_off_sink_faucet \
  output/20260720/083055_train_oattok_turn_off_sink_faucet_N200/checkpoints

# --- models ---
huggingface-cli download hackhackhack66666/rc-model-two-last \
  --repo-type model --local-dir /tmp/rc_model_two_last
cp /tmp/rc_model_two_last/policies/robocasa_turn_off_sink_faucet_topk_ep0500_sr0.580.ckpt my_models/
cp /tmp/rc_model_two_last/policies/robocasa_turn_off_sink_faucet_topk_lock.txt my_models/
cp /tmp/rc_model_two_last/tokenizers/sink_ep-3080_mse-0.002.ckpt \
  output/20260720/083055_train_oattok_turn_off_sink_faucet_N200/checkpoints/ep-3080_mse-0.002.ckpt

# --- dataset + Wave1 ---
huggingface-cli download hackhackhack66666/rc-last-two \
  --repo-type dataset --local-dir /tmp/rc_last_two
tar -xf /tmp/rc_last_two/turn_off_sink_faucet_N200.zarr.tar -C data/robocasa/
if [[ -d /tmp/rc_last_two/wave1_turn_off_sink_faucet ]]; then
  rsync -a /tmp/rc_last_two/wave1_turn_off_sink_faucet/ \
    output/eval/matched_s10000/robocasa/turn_off_sink_faucet/
fi
```

Verify:

```bash
test -f my_models/robocasa_turn_off_sink_faucet_topk_ep0500_sr0.580.ckpt
test -d data/robocasa/turn_off_sink_faucet_N200.zarr
test -f output/eval/matched_s10000/robocasa/turn_off_sink_faucet/baseline_seed10000/eval_log.json
# optional until Wave1 refresh:
# test -f output/eval/matched_s10000/robocasa/turn_off_sink_faucet/summary_literal5.json
```

---

## 3. Infra setup (match cluster)

### 3.1 Hardware / OS

| Need | Notes |
|------|--------|
| NVIDIA GPU | ≥12GB VRAM comfortable; BoN32 is slower, not much fatter on VRAM |
| RAM | ≥32GB recommended; collect with `n_workers>1` spikes hard |
| Disk | ≥25GB free (zarr 1.2G + ckpts + seed dirs + AWR npz) |
| CUDA | cluster uses **torch 2.5.1+cu124**; match if possible |
| Display | headless: `MUJOCO_GL=egl` |

### 3.2 Code

```bash
git clone https://github.com/Mirageinvo/OAT_BLT_research.git
cd OAT_BLT_research
git checkout robocasa
git submodule update --init --recursive   # if any
cd oat
```

### 3.3 MuJoCo 2.1.0

```bash
# place mujoco210 under ~/.mujoco/mujoco210  (same as cluster)
export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=egl
# after CUDA_VISIBLE_DEVICES remaps a single GPU, keep:
export MUJOCO_EGL_DEVICE_ID=0
```

### 3.4 Isolated RoboCasa venv (mandatory)

**Never** install RoboCasa into the shared MetaWorld/LIBERO `.venv` (robosuite 1.4).

```bash
cd oat
bash scripts/cluster_setup_robocasa_venv.sh
# creates .venv_robocasa (robosuite 1.5.1 + robocasa v0.2 + torch cu124 + numpy pin)

source .venv_robocasa/bin/activate
export OAT_USE_UV_RUN=0
bash scripts/patch_robosuite_egl_assert.sh || true
```

Smoke (optional, ~minutes):

```bash
MUJOCO_GL=egl python scripts/eval_policy_sim.py \
  -c my_models/robocasa_turn_off_sink_faucet_topk_ep0500_sr0.580.ckpt \
  -o /tmp/rc_sink_smoke \
  -n 1 --n_test 2 --test_start_seed 10000 \
  --use_k_tokens 8 --entropy_threshold 0 --temperature 1.0 --topk 10
```

---

## 4. Protocol differences vs MetaWorld (print before running)

| Topic | MetaWorld Table P habit | RoboCasa sink (you) |
|-------|-------------------------|---------------------|
| Variance | 5× `-n` on seed block 10000 | **5 distinct seeds**, one shot each |
| Uncertainty | often SD of `-n` | **SEM** over seeds (primary) |
| Δ | method − baseline (same layout) | same, but **SEM_Δ = sqrt(SEM_m²+SEM_b²)** unpaired |
| BoN N-sweep | `-n 5` dirs `bon_n16_n5/` | dirs `bon_n16_seed{10000..10004}/` |
| AWR source | often BoN8 collect, 30 or 100 ep | **BoN16 collect**, **100 ep only** |
| AWR OUT name | `awr_n5/` | `awr_bon16_seed{seed}/` |
| Aggregator | `summary.json` / matched triplet | `aggregate_robocasa_literal5.py` → `summary_literal5.json` |

Reference Wave1 sink baseline mean: **0.524**.

---

## 5. Job A — BoN16 + BoN32 (literal-5)

Wall-clock hint (1×GPU): ~**2–2.5 h / seed / N** under load →  
BoN16 ≈ **10–12 h**; BoN32 ≈ **same or slightly more**. Parallelize seeds across machines **only if** each uses a **disjoint** seed list.

```bash
cd oat
source .venv_robocasa/bin/activate
export OAT_USE_UV_RUN=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
export WANDB_MODE=disabled PYTHONUNBUFFERED=1

BASE=my_models/robocasa_turn_off_sink_faucet_topk_ep0500_sr0.580.ckpt
OUT=output/eval/matched_s10000/robocasa/turn_off_sink_faucet
GPU=0   # change per machine
export CUDA_VISIBLE_DEVICES=$GPU

run_bon () {
  local N="$1" seed="$2"
  local method="bon_n${N}"
  local out="${OUT}/${method}_seed${seed}"
  if [[ -f "${out}/eval_log.json" ]]; then
    echo "[skip] ${out}"
    return 0
  fi
  echo "[run] ${method} seed=${seed}"
  python scripts/eval_policy_sim.py \
    -c "${BASE}" \
    -o "${out}" \
    -n 1 --n_test 50 --test_start_seed "${seed}" \
    --use_k_tokens 8 --entropy_threshold 0 \
    --temperature 1.0 --topk 10 \
    --bon_free "${N}" --bon_signal vote
}

for N in 16 32; do
  for seed in 10000 10001 10002 10003 10004; do
    run_bon "$N" "$seed"
  done
done

python scripts/aggregate_robocasa_literal5.py --root "${OUT}"
```

**Parallel handoff example (2 GPUs / 2 people):**

- Person A: seeds `10000 10001 10002` for N=16 then N=32  
- Person B: seeds `10003 10004` for N=16 then N=32  
- Merge folders → one `aggregate_robocasa_literal5.py` run

✅ Each finished seed must have `eval_log.json` with `mean_success_rate_mean`.  
❌ Do not delete Wave1 `baseline_*` / `bon_n8_*` dirs.

---

## 6. Job B — AWR (BoN16-distill, 100 epochs)

Uses `scripts/cluster_robocasa_literal5_wave2_awr.sh` with overrides.

| Knob | Value |
|------|--------|
| `BON_N` | **16** (not 8) |
| `EPOCHS` | **100** (script refuses other values) |
| `COLLECT_SEED` | **0** |
| `N_CHUNKS` | **20000** |
| `beta` / `beta_kl` | **0.5** / **0.05** |
| Eval | literal-5, single-sample → folders `awr_bon16_seed*` |

```bash
cd oat
source .venv_robocasa/bin/activate
export OAT_USE_UV_RUN=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"

SUITE=turn_off_sink_faucet \
BASE_CKPT=my_models/robocasa_turn_off_sink_faucet_topk_ep0500_sr0.580.ckpt \
GPU=0 \
BON_N=16 \
EPOCHS=100 \
N_CHUNKS=20000 \
N_WORKERS=2 \
bash scripts/cluster_robocasa_literal5_wave2_awr.sh
```

Defaults when `BON_N=16`:
- dataset → `my_datasets/awr_s10000_robocasa_turn_off_sink_faucet_bon16.npz`
- ckpt → `my_models/awr_s10000_robocasa_turn_off_sink_faucet_bon16_e100.ckpt`
- eval dirs → `awr_bon16_seed{10000..10004}/`

Validate collect before/after (recommended):

```bash
python scripts/validate_awr.py \
  -i my_datasets/awr_s10000_robocasa_turn_off_sink_faucet_bon16.npz
```

---

## 7. Deliverables (send back)

```
output/eval/matched_s10000/robocasa/turn_off_sink_faucet/
  bon_n16_seed{10000..10004}/eval_log.json
  bon_n32_seed{10000..10004}/eval_log.json
  awr_bon16_seed{10000..10004}/eval_log.json
  summary_literal5.json
  wave2_awr_literal5.log                       # if present

my_datasets/awr_s10000_robocasa_turn_off_sink_faucet_bon16.npz
my_models/awr_s10000_robocasa_turn_off_sink_faucet_bon16_e100.ckpt
```

Plus: GPU model, torch/CUDA, git commit SHA, wall times.

---

## 8. Checklist before claiming numbers

- [ ] Branch `robocasa`, `.venv_robocasa`, `MUJOCO_GL=egl`
- [ ] Same `BASE_CKPT` path string in every eval log
- [ ] Every report seed ∈ `{10000…10004}`; `-n 1`; `n_test=50`
- [ ] BoN uses `--bon_signal vote`; AWR eval has **no** `--bon_free`
- [ ] AWR collect `--seed 0`, `--bon_n 16`, train `--epochs 100`
- [ ] `summary_literal5.json` has baseline + bon_n16 + bon_n32 + awr(_bon16)
- [ ] Δ vs **Wave1 baseline** (same 5 seeds), not vs TopK@2000
- [ ] No overlap of collect/selection seeds into report claims

---

## 9. FAQ / failure modes

| Symptom | Fix |
|---------|-----|
| `robosuite` / EGL assert | `bash scripts/patch_robosuite_egl_assert.sh`; check `MUJOCO_EGL_DEVICE_ID=0` after `CUDA_VISIBLE_DEVICES` |
| Wrong SR scale / crash on obs | Using shared `.venv` instead of `.venv_robocasa` |
| Wave2 refuses to start | Missing `baseline_seed*/eval_log.json` — restore Wave1 from HF |
| `EPOCHS!=100` error | Do not override; paper lock |
| OOM during collect | `N_WORKERS=1`; close other sims |
| Accidental `-n 5` | **Invalid** for RoboCasa paper — delete those outs, re-run literal-5 |

Canonical detail: `ROBOCASA.md` §0–§5, §7. Prefer **this guide** for the mentee track if a script disagrees on sink **Bon16-AWR**.
