# Agent guide — RoboCasa `turn_off_microwave`: BoN16/32 + AWR (literal-5)

Literal runbook for a mentee on their own GPU. Same protocol as coffee/close; this track is **turn_off_microwave**.

**Canonical protocol:** `ROBOCASA.md` (branch `robocasa`).  
**This guide** = microwave-only handoff: **BoN N∈{16,32}** eval + **AWR distill from BoN16 @ 100 epochs**.

Repo: [`Mirageinvo/OAT_BLT_research`](https://github.com/Mirageinvo/OAT_BLT_research) · branch **`robocasa`**.

---

## 0. Hard constraints (read first)

| Rule | RoboCasa microwave (THIS job) | MetaWorld / RoboMimic (DO NOT copy) |
|------|-------------------------------|--------------------------------------|
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
`my_models/robocasa_turn_off_microwave_topk_ep0500_sr0.620.ckpt`  
(TopK @ selection seed 2000, SR=0.620, epoch 500. Do **not** retrain / swap.)

**Already done on cluster (do not redo unless asked):** Wave1 **baseline** (+ **BoN8** when present on HF)  
→ pull from HF (below). Your job starts at **BoN16 / BoN32** (+ AWR).

**Wave1 note:** baseline / BoN8 may still be landing on HF after cluster finishes. Seed10000 baseline sample was **0.46**; full 5-seed mean arrives with `summary_literal5.json`. **Do not start AWR** until all `baseline_seed{10000..10004}/eval_log.json` exist locally.

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
❌ Train AWR with `--epochs 30` for paper microwave.

---

## 2. What to download from HuggingFace

### 2.1 Repo (shared pack for sink + microwave)

| Role | HF repo | Type |
|------|---------|------|
| **Models + zarr + Wave1** | [`hackhackhack66666/rc-last-two`](https://huggingface.co/datasets/hackhackhack66666/rc-last-two) | dataset |

Videos under `media/` are **optional**. Paper needs `eval_log.json` only.

### 2.2 Layout (`rc-last-two`)

```
policies/
  robocasa_turn_off_microwave_topk_ep0500_sr0.620.ckpt     # ~893M
  robocasa_turn_off_microwave_topk_lock.txt
  robocasa_turn_off_sink_faucet_topk_ep0500_sr0.580.ckpt   # (other task)
  robocasa_turn_off_sink_faucet_topk_lock.txt
tokenizers/
  microwave_ep-2720_mse-0.002.ckpt                         # ~89M
  sink_ep-3080_mse-0.002.ckpt
turn_off_microwave_N200.zarr/                              # ~1.2G
  ROBOCASA_SOURCE.txt
turn_off_sink_faucet_N200.zarr/
wave1_turn_off_microwave/                                  # when uploaded
  summary_literal5.json
  baseline_seed{10000..10004}/eval_log.json
  bon_n8_seed{10000..10004}/eval_log.json
wave1_turn_off_sink_faucet/
README.md
```

### 2.3 Mentee: download + place under `oat/`

```bash
cd oat
mkdir -p data/robocasa my_models my_datasets \
  output/eval/matched_s10000/robocasa/turn_off_microwave \
  output/20260720/061925_train_oattok_turn_off_microwave_N200/checkpoints

huggingface-cli download hackhackhack66666/rc-last-two \
  --repo-type dataset --local-dir /tmp/rc_last_two
# or: git xet install && git clone https://huggingface.co/datasets/hackhackhack66666/rc-last-two /tmp/rc_last_two

cp /tmp/rc_last_two/policies/robocasa_turn_off_microwave_topk_ep0500_sr0.620.ckpt my_models/
cp /tmp/rc_last_two/policies/robocasa_turn_off_microwave_topk_lock.txt my_models/
cp /tmp/rc_last_two/tokenizers/microwave_ep-2720_mse-0.002.ckpt \
  output/20260720/061925_train_oattok_turn_off_microwave_N200/checkpoints/ep-2720_mse-0.002.ckpt

rsync -a /tmp/rc_last_two/turn_off_microwave_N200.zarr data/robocasa/
if [[ -d /tmp/rc_last_two/wave1_turn_off_microwave ]]; then
  rsync -a /tmp/rc_last_two/wave1_turn_off_microwave/ \
    output/eval/matched_s10000/robocasa/turn_off_microwave/
fi
```

Verify:

```bash
test -f my_models/robocasa_turn_off_microwave_topk_ep0500_sr0.620.ckpt
test -d data/robocasa/turn_off_microwave_N200.zarr
test -f output/eval/matched_s10000/robocasa/turn_off_microwave/baseline_seed10000/eval_log.json
# optional until Wave1 refresh:
# test -f output/eval/matched_s10000/robocasa/turn_off_microwave/summary_literal5.json
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
export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
```

### 3.4 Isolated RoboCasa venv (mandatory)

```bash
cd oat
bash scripts/cluster_setup_robocasa_venv.sh
source .venv_robocasa/bin/activate
export OAT_USE_UV_RUN=0
bash scripts/patch_robosuite_egl_assert.sh || true
```

Smoke (optional):

```bash
MUJOCO_GL=egl python scripts/eval_policy_sim.py \
  -c my_models/robocasa_turn_off_microwave_topk_ep0500_sr0.620.ckpt \
  -o /tmp/rc_mw_smoke \
  -n 1 --n_test 2 --test_start_seed 10000 \
  --use_k_tokens 8 --entropy_threshold 0 --temperature 1.0 --topk 10
```

---

## 4. Protocol differences vs MetaWorld (print before running)

| Topic | MetaWorld Table P habit | RoboCasa microwave (you) |
|-------|-------------------------|--------------------------|
| Variance | 5× `-n` on seed block 10000 | **5 distinct seeds**, one shot each |
| Uncertainty | often SD of `-n` | **SEM** over seeds (primary) |
| Δ | method − baseline | **SEM_Δ = sqrt(SEM_m²+SEM_b²)** unpaired |
| BoN N-sweep | `-n 5` dirs | `bon_n16_seed{10000..10004}/` |
| AWR source | often BoN8 / 30 ep | **BoN16 collect**, **100 ep only** |
| Aggregator | matched triplet | `aggregate_robocasa_literal5.py` |

---

## 5. Job A — BoN16 + BoN32 (literal-5)

~**2–2.5 h / seed / N** → BoN16 ≈ **10–12 h**; BoN32 similar. Disjoint seed lists only if parallelizing.

```bash
cd oat
source .venv_robocasa/bin/activate
export OAT_USE_UV_RUN=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
export WANDB_MODE=disabled PYTHONUNBUFFERED=1

BASE=my_models/robocasa_turn_off_microwave_topk_ep0500_sr0.620.ckpt
OUT=output/eval/matched_s10000/robocasa/turn_off_microwave
GPU=0
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

✅ `eval_log.json` with `mean_success_rate_mean` per seed.  
❌ Do not delete Wave1 `baseline_*` / `bon_n8_*`.

---

## 6. Job B — AWR (BoN16-distill, 100 epochs)

| Knob | Value |
|------|--------|
| `BON_N` | **16** |
| `EPOCHS` | **100** |
| `COLLECT_SEED` | **0** |
| `N_CHUNKS` | **20000** |
| `beta` / `beta_kl` | **0.5** / **0.05** |

```bash
cd oat
source .venv_robocasa/bin/activate
export OAT_USE_UV_RUN=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"

SUITE=turn_off_microwave \
BASE_CKPT=my_models/robocasa_turn_off_microwave_topk_ep0500_sr0.620.ckpt \
GPU=0 \
BON_N=16 \
EPOCHS=100 \
N_CHUNKS=20000 \
N_WORKERS=2 \
bash scripts/cluster_robocasa_literal5_wave2_awr.sh
```

- dataset → `my_datasets/awr_s10000_robocasa_turn_off_microwave_bon16.npz`
- ckpt → `my_models/awr_s10000_robocasa_turn_off_microwave_bon16_e100.ckpt`
- eval → `awr_bon16_seed{10000..10004}/`

```bash
python scripts/validate_awr.py \
  -i my_datasets/awr_s10000_robocasa_turn_off_microwave_bon16.npz
```

---

## 7. Deliverables (send back)

```
output/eval/matched_s10000/robocasa/turn_off_microwave/
  bon_n16_seed{10000..10004}/eval_log.json
  bon_n32_seed{10000..10004}/eval_log.json
  awr_bon16_seed{10000..10004}/eval_log.json
  summary_literal5.json
  wave2_awr_literal5.log

my_datasets/awr_s10000_robocasa_turn_off_microwave_bon16.npz
my_models/awr_s10000_robocasa_turn_off_microwave_bon16_e100.ckpt
```

Plus: GPU / torch / CUDA / git SHA / wall times.

---

## 8. Checklist before claiming numbers

- [ ] Branch `robocasa`, `.venv_robocasa`, `MUJOCO_GL=egl`
- [ ] Same `BASE_CKPT` in every eval log
- [ ] Seeds `{10000…10004}`; `-n 1`; `n_test=50`
- [ ] BoN `--bon_signal vote`; AWR eval **no** `--bon_free`
- [ ] AWR collect `--seed 0`, `--bon_n 16`, train `--epochs 100`
- [ ] Δ vs **Wave1 baseline**, not TopK@2000

---

## 9. FAQ / failure modes

| Symptom | Fix |
|---------|-----|
| EGL / robosuite assert | `patch_robosuite_egl_assert.sh`; `MUJOCO_EGL_DEVICE_ID=0` |
| Wrong obs / crash | shared `.venv` instead of `.venv_robocasa` |
| Wave2 refuses | missing baseline logs — restore Wave1 from HF |
| `EPOCHS!=100` | paper lock — do not override |
| `-n 5` | invalid for RoboCasa paper |

Canonical: `ROBOCASA.md`. Prefer **this guide** for mentee microwave **Bon16-AWR**.
