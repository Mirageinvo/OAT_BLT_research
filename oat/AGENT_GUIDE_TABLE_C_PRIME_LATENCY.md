# Agent guide — Table C′ latency (fair-KV)

Literal runbook for an agent measuring **policy-forward latency** exactly as **Table C′** in [`RESULTS.md`](RESULTS.md) (§ Table C′ — fair-KV rebuttal).

**This is Table C′ only** (`latency_fair_kv.json`).  
It does **not** replace main **Table C** (`latency.json`, deployed Single/AWR without KV). Do not mix protocols.

Canon summary (locked numbers): `output/eval/matched_s10000/table_c_fair_kv.json` (`fair_kv: true`, `paper_locked: true`).

---

## 0. Hard constraints (read first)

| Rule | Detail |
|------|--------|
| **What** | ms / **policy call**, batch=1 — **not** MuJoCo / render / episode wall-clock |
| **Modes** | Single + BoN N=8 vote + AWR — **three** timings per suite |
| **Fair-KV paths** | Single & AWR = `predict_action` (**KV-cache**). BoN = `predict_action_bon_free` / `generate` (**KV**). |
| **Paper number** | **mean ± std of per-trial medians** (`trial_median_mean_ms` ± `trial_median_std_ms`), **not** a single run’s raw mean |
| **Trials × reps** | **`TRIALS=8`**, **`REPS=10`**, **`WARMUP=20`** (warmup **excluded** from stats) |
| **Obs** | val dataset from checkpoint Hydra cfg, **batch=1**; obs counter **reset + warmup each trial** |
| **HW** | same stack as paper SR: cluster **Tesla V100-SXM2-32GB**, torch **2.5.1+cu124**, `MUJOCO_GL=egl`, same docker as matched eval |
| **Ckpts** | Base = Wave1 `base_ckpt` from suite summary; AWR = **`my_models/awr_s10000_<suite>.ckpt`** (same as Wave2 `awr_n5/`). ❌ other AWR / exploratory / wrong epoch |
| **SR** | **Do not** re-run Table P for latency. SR columns stay from Table P |
| **Git** | `OAT_GIT_COMMIT` **required** (docker often has no `.git`). Refuse run without it |
| **Overwrite** | Locked C′ is **PAPER-LOCKED**. Overwrite only if the user **explicitly** asks to remasure. Prefer new out-dir or confirm first |

### Forbidden (reject / abort)

- Table C deployed path (`--deployed` / `FAIR_KV=0`) when asked for **C′**
- `my_scripts/measure_latency*.py` / adaptive latency scripts for paper C′
- Dummy random obs (must be val pipeline)
- `TRIALS=1` for paper C′ (script defaults to 8 when `FAIR_KV=1`, but do **not** override to 1)
- Mac / laptop / different GPU than paper SR machine
- Mixing sim wall-clock into “policy latency”
- Timing a different AWR ckpt than Wave2 Table P
- Rewriting `latency.json` (Table C) when running C′
- Claiming Δ(BoN−Single) as “faster” if |Δ| ≲ trial std (noise)

---

## 1. What each mode times (must match code)

Implemented only in `scripts/measure_latency_paper.py --fair_kv`:

| Mode | Call | Kwargs (fixed) | Ckpt |
|------|------|----------------|------|
| **single** | `policy.predict_action(...)` | `use_k_tokens=8`, `temperature=1.0`, `topk=10` | Wave1 **base** |
| **bon** | `policy.predict_action_adaptive(..., bon_free=8)` | `use_k_tokens=8`, `entropy_threshold=0`, `temperature=1.0`, `topk=10`, **vote** | same **base** |
| **awr** | `awr.predict_action(...)` | same as single | `my_models/awr_s10000_<suite>.ckpt` |

Deployed Table C uses `predict_action_adaptive` for Single/AWR (no KV) — **wrong for C′**.

---

## 2. Suites (C′ locked set)

Exactly these seven (same as `build_table_c.py` / locked table):

```text
can  coffee-pull  stick-pull  disassemble  box-close  square  lift
```

Per-suite artifacts:

| Suite | Output |
|-------|--------|
| each | `output/eval/matched_s10000/<suite>/latency_fair_kv.json` |
| aggregate | `output/eval/matched_s10000/table_c_fair_kv.json` |

Ckpt resolution: script reads matched summary for `base_ckpt`; AWR path = `my_models/awr_s10000_<suite>.ckpt`. Both files must exist before start.

---

## 3. Environment

```bash
# on cluster host (paths may match your layout)
cd ~/mipt_paper/oat   # or docker workdir /workspace/oat

# git provenance — REQUIRED for paper-proof
export OAT_GIT_COMMIT="$(git -C ~/mipt_paper rev-parse HEAD)"
export OAT_GIT_BRANCH="$(git -C ~/mipt_paper rev-parse --abbrev-ref HEAD)"
export OAT_GIT_DIRTY="$(git -C ~/mipt_paper status --porcelain | grep -q . && echo 1 || echo 0)"

# inside docker / venv used for paper MW+RM eval (.venv, not .venv_robocasa)
source .venv/bin/activate   # or: docker exec … with .venv
export MUJOCO_GL=egl
export OAT_USE_UV_RUN=0
```

Use **one free GPU** (V100). Do not share with heavy train/eval if it perturbs timing; idle GPU preferred.

---

## 4. Canonical launch (copy-paste)

**Only** this entrypoint:

```bash
cd /workspace/oat   # inside paper docker

FAIR_KV=1 \
TRIALS=8 \
REPS=10 \
WARMUP=20 \
GPU=0 \
SUITES="can coffee-pull stick-pull disassemble box-close square lift" \
OAT_GIT_COMMIT="${OAT_GIT_COMMIT}" \
OAT_GIT_BRANCH="${OAT_GIT_BRANCH}" \
OAT_GIT_DIRTY="${OAT_GIT_DIRTY}" \
bash scripts/cluster_latency_paper_done.sh
```

What the script does:

1. `source scripts/cluster_gpu_env.sh $GPU` + EGL patch  
2. For each suite: `python scripts/measure_latency_paper.py --suite … --fair_kv --reps 10 --trials 8 --warmup 20`  
3. Writes `…/<suite>/latency_fair_kv.json` (**does not** touch `latency.json`)  
4. Runs `python scripts/build_table_c.py --fair_kv` → `table_c_fair_kv.json`

Log: `logs/latency_fair_kv_s10000_gpu${GPU}.log`

### Single-suite remasure (debug only)

```bash
source scripts/cluster_gpu_env.sh 0
python scripts/measure_latency_paper.py \
  --suite box-close \
  --fair_kv \
  -d "${OAT_DEVICE}" \
  --reps 10 --trials 8 --warmup 20
# then rebuild aggregate if needed:
python scripts/build_table_c.py --fair_kv
```

---

## 5. Acceptance checklist (before reporting)

For **each** suite `latency_fair_kv.json`:

- [ ] `"fair_kv": true`
- [ ] `"paper_proof": false` (C′ is rebuttal; Table C is `paper_proof`)
- [ ] `protocol.n_trials == 8`, `protocol.reps == 10`, `protocol.warmup == 20`, `protocol.batch_size == 1`
- [ ] `protocol.single_path` / `awr_path` contain **`predict_action (KV)`**
- [ ] `protocol.bon_path` is BoN / generate (KV)
- [ ] `git_commit` non-empty; `gpu_name` = Tesla V100…; torch ≈ `2.5.1+cu124`
- [ ] `base_ckpt` / `awr_ckpt` match Wave1 / Wave2 Table P paths
- [ ] Each mode has `n_trials=8`, `trial_medians_ms` length 8; paper cell uses **`median_ms` = mean of trial medians** and **`std_ms` = std of trial medians**
- [ ] Aggregate `table_c_fair_kv.json` rebuilt and lists all requested suites

### How to quote a cell

From RESULTS Table C′ style:

```text
Single / BoN / AWR  =  mean±std of per-trial medians  (ms)
BoN−Single          =  difference of those means
```

Do **not** interpret |BoN−Single| ≲ 3 ms / within std as a real speedup (locked read: overhead within noise).

---

## 6. Relation to Table C (do not confuse)

| | Table C (main) | Table C′ (this guide) |
|--|----------------|------------------------|
| Flag | `FAIR_KV=0` (default) | **`FAIR_KV=1` / `--fair_kv`** |
| Artifact | `latency.json` | **`latency_fair_kv.json`** |
| Single/AWR | `predict_action_adaptive` (no KV) | **`predict_action` (KV)** |
| Trials (paper) | often 1 trial × 10 reps → median | **8 trials × 10 reps → mean±std of medians** |
| Role in paper | main latency table | appendix / fair-KV rebuttal |

If the user says “latency like the paper main table” → Table **C**, not this guide.  
If they say **C′ / fair-KV / apples-to-apples KV** → **this** guide.

---

## 7. Files (source of truth)

| File | Role |
|------|------|
| `scripts/measure_latency_paper.py` | timing + JSON schema |
| `scripts/cluster_latency_paper_done.sh` | multi-suite launcher (`FAIR_KV=1` → trials=8) |
| `scripts/build_table_c.py --fair_kv` | aggregate |
| `RESULTS.md` § Table C′ | locked numbers + narrative |
| `output/eval/matched_s10000/table_c_fair_kv.json` | locked aggregate |

---

## 8. Done criteria

1. All requested suites have fresh or verified `latency_fair_kv.json` matching §5.  
2. `table_c_fair_kv.json` rebuilt.  
3. Report a markdown table: Suite | Single | BoN8 | AWR | BoN−Single (ms), plus git commit + GPU + torch.  
4. Explicitly state: **policy-forward only; SR unchanged; C′ ≠ Table C.**
