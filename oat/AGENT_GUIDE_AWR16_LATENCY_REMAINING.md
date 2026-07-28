# Agent / mentee guide — remaining **AWR16** latency (Table C′)

Fill the empty **AWR16 (KV)** cells in [`RESULTS.md`](RESULTS.md) § Table C′.

**Already DONE (do not remasure):** RoboMimic `can` / `lift` / `square` + RoboCasa `coffee_press_button`.  
**Still open:** MetaWorld ×4 + RoboCasa `close_drawer` / `turn_off_sink_faucet` / `turn_off_microwave`.

This guide is **latency only** (policy-forward ms). It does **not** train AWR, does **not** run Table P SR, does **not** touch locked `latency_fair_kv.json` (Single/BoN8/AWR8).

Full C′ protocol background: [`AGENT_GUIDE_TABLE_C_PRIME_LATENCY.md`](AGENT_GUIDE_TABLE_C_PRIME_LATENCY.md).

---

## 0. Hard rules

| Rule | Detail |
|------|--------|
| **What** | ms / **policy call**, batch=1 — **not** MuJoCo / render / episode wall-clock |
| **Mode timed** | **only** `awr16` = `predict_action` (**KV-cache**) on the **BoN16-distill @100ep** ckpt |
| **Do not retime** | Single / BoN8 / BoN16 / BoN32 / AWR8 — already locked or filled |
| **Paper number** | **mean ± std of per-trial medians** (`trial_median_mean_ms` ± `trial_median_std_ms`) |
| **Trials × reps** | **`TRIALS=8`**, **`REPS=10`**, **`WARMUP=20`** (warmup excluded) |
| **Flags** | `--fair_kv --skip_awr --no_single --bon_ns "" --awr16_ckpt <path>` |
| **Output** | merges into `latency_fair_kv_n16.json` (**never** overwrite `latency_fair_kv.json`) |
| **HW** | cluster **Tesla V100**, same docker as paper eval, `MUJOCO_GL=egl` |
| **Git** | `OAT_GIT_COMMIT` **required** |
| **Disk** | ckpts ~0.9 GB; host often near full → **download ONE → measure → DELETE** |

### Forbidden

- Remasuring Single/BoN*/AWR8 “for completeness”
- Writing into locked `latency_fair_kv.json` / `table_c_fair_kv.json` without explicit remasure ask
- Using exploratory `policy_awr_*` or AWR**8** Wave2 ckpt as AWR16
- `TRIALS=1` / dummy obs / laptop GPU
- Claiming AWR16 is “faster” if |Δ| ≲ trial std

---

## 1. Remaining suites (checklist)

| Suite | Family | Expected HF filename *(when uploaded)* | Local ckpt fallback | `latency_fair_kv_n16.json` path | Venv |
|-------|--------|----------------------------------------|---------------------|----------------------------------|------|
| coffee-pull | MetaWorld | `metaworld_coffee-pull_awr_bon16_e100.ckpt` *(or similar — check HF)* | Wave2 Bon16-AWR @100ep local path | `output/eval/matched_s10000/coffee-pull/latency_fair_kv_n16.json` | `.venv` |
| stick-pull | MetaWorld | `metaworld_stick-pull_awr_bon16_e100.ckpt` | idem | `…/stick-pull/latency_fair_kv_n16.json` | `.venv` |
| disassemble | MetaWorld | `metaworld_disassemble_awr_bon16_e100.ckpt` | idem | `…/disassemble/latency_fair_kv_n16.json` | `.venv` |
| box-close | MetaWorld | `metaworld_box-close_awr_bon16_e100.ckpt` | idem | `…/box-close/latency_fair_kv_n16.json` | `.venv` |
| close_drawer | RoboCasa | `robocasa_close_drawer_awr_bon16_e100.ckpt` | `my_models/awr_s10000_robocasa_close_drawer_bon16_e100.ckpt` *(after Wave2)* | `…/robocasa/close_drawer/latency_fair_kv_n16.json` | **`.venv_robocasa`** |
| turn_off_sink_faucet | RoboCasa | `robocasa_turn_off_sink_faucet_awr_bon16_e100.ckpt` | idem pattern | `…/robocasa/turn_off_sink_faucet/latency_fair_kv_n16.json` | **`.venv_robocasa`** |
| turn_off_microwave | RoboCasa | `robocasa_turn_off_microwave_awr_bon16_e100.ckpt` | idem pattern | `…/robocasa/turn_off_microwave/latency_fair_kv_n16.json` | **`.venv_robocasa`** |

**HF status (2026-07-28):** [`Mirageinv/AWR`](https://huggingface.co/Mirageinv/AWR) has **only**:

```text
robomimic_can_awr_bon16_e100.ckpt
robomimic_lift_awr_bon16_e100.ckpt
robomimic_square_awr_bon16_e100.ckpt
robocasa_coffee_press_button_awr_bon16_e100.ckpt
```

→ remaining suites are **blocked until** mentee uploads Bon16-AWR@100ep ckpts to HF **or** you have a local Wave2 Bon16-AWR checkpoint.

**Verify HF before starting:**

```bash
hf api model info Mirageinv/AWR --json | python3 -c \
  'import sys,json; print("\n".join(sorted(s["rfilename"] for s in json.load(sys.stdin)["siblings"])))'
```

---

## 2. What each call times

| Mode | Call | Kwargs | Ckpt |
|------|------|--------|------|
| **awr16** | `policy.predict_action(...)` | `use_k_tokens=8`, `temperature=1.0`, `topk=10` | Bon16-distill **e100** AWR |

Obs: val pipeline from **that** ckpt’s Hydra cfg, batch=1; counter reset + warmup **each** of 8 trials.

---

## 3. Environment (cluster)

```bash
# SSH (example)
ssh -i ~/.ssh/mipt_lab -o IdentitiesOnly=yes askhabaliev_gs@100.98.148.137
docker exec -it oat_mw_bon32_fix bash   # or the paper RM docker if MW-only

cd /workspace/oat

# REQUIRED provenance (docker often has no .git)
export OAT_GIT_COMMIT="$(git -C /path/to/mipt_paper rev-parse HEAD 2>/dev/null || echo UNKNOWN)"
# Prefer real commit from host repo:
#   export OAT_GIT_COMMIT=$(git -C ~/mipt_paper rev-parse HEAD)
export OAT_GIT_BRANCH=robocasa
export OAT_GIT_DIRTY=1
export MUJOCO_GL=egl
export OAT_USE_UV_RUN=0

# free GPU (do not share with Wave1 BoN if it perturbs timing)
export GPU=0   # or 1
source scripts/cluster_gpu_env.sh "${GPU}"
bash scripts/patch_robosuite_egl_assert.sh || true

df -h /workspace/oat /tmp   # need ≳2G free for one ckpt + temp
```

**Python:**

| Suites | Interpreter |
|--------|-------------|
| MetaWorld / RoboMimic | `.venv/bin/python` (if broken: `_runtime_uv_python/.../python3.10` + `.venv` site-packages — see `cluster_latency_awr16_hf.sh`) |
| RoboCasa | **`.venv_robocasa/bin/python`** (robosuite 1.5) |

**`hf` CLI:** use `/home/askhabaliev_gs/.local/bin/hf` (avoid broken `.venv/bin/hf`).

---

## 4. Canonical one-suite command (copy-paste)

### 4a. From HuggingFace (preferred when file exists)

```bash
SUITE=close_drawer                    # CHANGE
HF_FILE=robocasa_close_drawer_awr_bon16_e100.ckpt   # CHANGE — exact name from HF list
VENV=.venv_robocasa                   # .venv for MW
STAGE=/tmp/awr16_stage
mkdir -p "${STAGE}" logs
rm -rf "${STAGE:?}"/*

# download ONE file
PATH="$(echo "$PATH" | tr ':' '\n' | grep -v '/\.venv' | paste -sd: -)" \
  /home/askhabaliev_gs/.local/bin/hf download Mirageinv/AWR "${HF_FILE}" --local-dir "${STAGE}"
CKPT="${STAGE}/${HF_FILE}"
test -f "${CKPT}"

# activate venv
source "${VENV}/bin/activate"   # or runtime-python fallback from scripts/cluster_latency_awr16_hf.sh

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl \
OAT_GIT_COMMIT="${OAT_GIT_COMMIT}" OAT_GIT_BRANCH="${OAT_GIT_BRANCH}" OAT_GIT_DIRTY="${OAT_GIT_DIRTY}" \
python scripts/measure_latency_paper.py \
  --suite "${SUITE}" \
  -d "${OAT_DEVICE}" \
  --reps 10 --trials 8 --warmup 20 \
  --fair_kv \
  --skip_awr \
  --no_single \
  --bon_ns "" \
  --awr16_ckpt "${CKPT}" \
  2>&1 | tee -a "logs/latency_awr16_${SUITE}_gpu${GPU}.log"

# ALWAYS delete after measure
rm -f "${CKPT}"
rm -rf "${STAGE:?}"/*
df -h /workspace/oat
```

### 4b. From local Wave2 ckpt (no HF)

```bash
SUITE=coffee-pull
CKPT=my_models/awr_s10000_coffee-pull_bon16_e100.ckpt   # exact local path after Wave2
# same measure_latency_paper.py flags as above with --awr16_ckpt "${CKPT}"
# do NOT delete the Wave2 artifact unless disk-critical (prefer copy to /tmp then delete copy)
```

### 4c. Batch launcher (extend when HF grows)

Existing script covers only the **already-done** HF set:

```bash
OAT_GIT_COMMIT=... GPU=0 bash scripts/cluster_latency_awr16_hf.sh
```

To measure remaining suites after upload: **edit** `ALL_JOBS=(...)` in `scripts/cluster_latency_awr16_hf.sh` — add lines:

```bash
"coffee-pull|<exact_hf_filename>|.venv"
"stick-pull|<exact_hf_filename>|.venv"
"disassemble|<exact_hf_filename>|.venv"
"box-close|<exact_hf_filename>|.venv"
"close_drawer|robocasa_close_drawer_awr_bon16_e100.ckpt|.venv_robocasa"
"turn_off_sink_faucet|robocasa_turn_off_sink_faucet_awr_bon16_e100.ckpt|.venv_robocasa"
"turn_off_microwave|robocasa_turn_off_microwave_awr_bon16_e100.ckpt|.venv_robocasa"
```

Script skips suites that already have `modes.awr16` in `latency_fair_kv_n16.json`.

---

## 5. Acceptance (per suite)

Open `…/latency_fair_kv_n16.json` and check:

- [ ] `"fair_kv": true`
- [ ] `modes.awr16` present
- [ ] `protocol.n_trials == 8`, `reps == 10`, `warmup == 20`, `batch_size == 1`
- [ ] `awr16` path is **`predict_action (KV)`**
- [ ] `n_trials=8`, `trial_medians_ms` length 8
- [ ] Paper cell = `trial_median_mean_ms` ± `trial_median_std_ms` (same as `median_ms`/`std_ms` after fair-kv bookkeeping)
- [ ] `git_commit` non-empty; GPU = V100
- [ ] Locked `latency_fair_kv.json` **unchanged** (mtime / Single/BoN8/AWR8 identical)

Quick check:

```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path("output/eval/matched_s10000/robocasa/close_drawer/latency_fair_kv_n16.json")  # CHANGE
d = json.loads(p.read_text())
m = d["modes"]["awr16"]
print(f"{m['trial_median_mean_ms']:.1f}±{m['trial_median_std_ms']:.1f}")
PY
```

---

## 6. Fill RESULTS.md Table C′

For each finished suite, set **AWR16 (KV)** cell to `**XX.X±Y.Y**` (1 decimal).

Leave MetaWorld/RC missing cells as `TBD` until measured.  
Update the Fill blurb under Table C′ to list newly completed suites.

Do **not** change Single / BoN8 / BoN16 / BoN32 / AWR8 columns.

---

## 7. Done criteria

1. All 7 remaining suites have `modes.awr16` in their `latency_fair_kv_n16.json`.  
2. Staging `/tmp/awr16_stage` empty; no orphan ~GB ckpts on `/workspace/oat`.  
3. `RESULTS.md` Table C′ AWR16 column filled for those suites.  
4. Report one line per suite: `suite  XX.X±Y.Y ms  git=<commit>  gpu=V100`.

---

## 8. Files

| File | Role |
|------|------|
| `scripts/measure_latency_paper.py` | `--fair_kv --awr16_ckpt …` |
| `scripts/cluster_latency_awr16_hf.sh` | disk-safe HF cycle (extend `ALL_JOBS`) |
| `AGENT_GUIDE_TABLE_C_PRIME_LATENCY.md` | full C′ protocol |
| `RESULTS.md` § Table C′ | numbers to fill |
| HF [`Mirageinv/AWR`](https://huggingface.co/Mirageinv/AWR) | Bon16-AWR@100ep ckpts |
