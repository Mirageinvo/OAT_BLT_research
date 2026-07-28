# Agent / mentee guide — remaining **RoboCasa AWR16** latency (Table C′)

Fill the empty **AWR16 (KV)** cells for the **three remaining RoboCasa** tasks in [`RESULTS.md`](RESULTS.md) § Table C′.

**Already DONE (do not remasure):**
- RoboMimic `can` / `lift` / `square` — AWR16 filled
- RoboCasa `coffee_press_button` — AWR16 filled

**Still open (this guide only):** RoboCasa `close_drawer` / `turn_off_sink_faucet` / `turn_off_microwave`.

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
| **HW** | cluster **Tesla V100**, docker `oat_mw_bon32_fix`, `MUJOCO_GL=egl` |
| **Git** | `OAT_GIT_COMMIT` **required** |
| **Disk** | ckpts ~0.9 GB; host often near full → **download ONE → measure → DELETE** |
| **Venv** | **always** `.venv_robocasa` (robosuite 1.5) |

### Forbidden

- Remasuring Single/BoN*/AWR8 “for completeness”
- Writing into locked `latency_fair_kv.json` / `table_c_fair_kv.json` without explicit remasure ask
- Using exploratory `policy_awr_*` or AWR**8** Wave2 ckpt as AWR16
- `TRIALS=1` / dummy obs / laptop GPU
- Claiming AWR16 is “faster” if |Δ| ≲ trial std

---

## 1. Remaining RoboCasa suites (checklist)

| Suite | Expected HF filename *(when uploaded)* | Local ckpt fallback *(after Wave2 Bon16-AWR)* | `latency_fair_kv_n16.json` |
|-------|----------------------------------------|-----------------------------------------------|----------------------------|
| `close_drawer` | `robocasa_close_drawer_awr_bon16_e100.ckpt` | `my_models/awr_s10000_robocasa_close_drawer_bon16_e100.ckpt` | `output/eval/matched_s10000/robocasa/close_drawer/latency_fair_kv_n16.json` |
| `turn_off_sink_faucet` | `robocasa_turn_off_sink_faucet_awr_bon16_e100.ckpt` | `my_models/awr_s10000_robocasa_turn_off_sink_faucet_bon16_e100.ckpt` | `…/robocasa/turn_off_sink_faucet/latency_fair_kv_n16.json` |
| `turn_off_microwave` | `robocasa_turn_off_microwave_awr_bon16_e100.ckpt` | `my_models/awr_s10000_robocasa_turn_off_microwave_bon16_e100.ckpt` | `…/robocasa/turn_off_microwave/latency_fair_kv_n16.json` |

**HF status (2026-07-28):** [`Mirageinv/AWR`](https://huggingface.co/Mirageinv/AWR) has RoboMimic + `coffee_press_button` only — **none of the three above yet**.

→ blocked until mentee uploads Bon16-AWR@100ep ckpts to HF **or** local Wave2 Bon16-AWR ckpt exists on cluster.

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
ssh -i ~/.ssh/mipt_lab -o IdentitiesOnly=yes askhabaliev_gs@100.98.148.137
docker exec -it oat_mw_bon32_fix bash

cd /workspace/oat

export OAT_GIT_COMMIT="$(git -C /home/askhabaliev_gs/mipt_paper rev-parse HEAD)"
export OAT_GIT_BRANCH=robocasa
export OAT_GIT_DIRTY=1
export MUJOCO_GL=egl
export OAT_USE_UV_RUN=0

export GPU=0   # or 1 — avoid sharing with heavy Wave1 eval if possible
source scripts/cluster_gpu_env.sh "${GPU}"
bash scripts/patch_robosuite_egl_assert.sh || true

df -h /workspace/oat /tmp   # need ≳2G free for one ckpt + temp
```

**Python:** **`.venv_robocasa/bin/python`** only.  
If broken shebang: `_runtime_uv_python/.../python3.10` + `.venv_robocasa` site-packages (see `cluster_latency_awr16_hf.sh`).

**`hf` CLI:** `/home/askhabaliev_gs/.local/bin/hf` on **host** (token in `~/.cache/huggingface/token`). Inside docker HF auth may fail — download on host to `/tmp/awr16_stage`, bind-mount or copy into container.

---

## 4. Canonical one-suite command (copy-paste)

### 4a. From HuggingFace (preferred when file exists)

```bash
SUITE=close_drawer                    # CHANGE: close_drawer | turn_off_sink_faucet | turn_off_microwave
HF_FILE=robocasa_close_drawer_awr_bon16_e100.ckpt   # CHANGE — exact name from HF
STAGE=/tmp/awr16_stage
mkdir -p "${STAGE}" logs
rm -rf "${STAGE:?}"/*

PATH="$(echo "$PATH" | tr ':' '\n' | grep -v '/\.venv' | paste -sd: -)" \
  /home/askhabaliev_gs/.local/bin/hf download Mirageinv/AWR "${HF_FILE}" --local-dir "${STAGE}"
CKPT="${STAGE}/${HF_FILE}"
test -f "${CKPT}"

source .venv_robocasa/bin/activate

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

rm -f "${CKPT}"
rm -rf "${STAGE:?}"/*
df -h /workspace/oat
```

### 4b. From local Wave2 ckpt (no HF)

```bash
SUITE=turn_off_sink_faucet
CKPT=my_models/awr_s10000_robocasa_turn_off_sink_faucet_bon16_e100.ckpt
# same measure_latency_paper.py flags as §4a with --awr16_ckpt "${CKPT}"
```

### 4c. Batch launcher (after HF upload)

Edit `ALL_JOBS=(...)` in `scripts/cluster_latency_awr16_hf.sh` — add only RC lines:

```bash
"close_drawer|robocasa_close_drawer_awr_bon16_e100.ckpt|.venv_robocasa"
"turn_off_sink_faucet|robocasa_turn_off_sink_faucet_awr_bon16_e100.ckpt|.venv_robocasa"
"turn_off_microwave|robocasa_turn_off_microwave_awr_bon16_e100.ckpt|.venv_robocasa"
```

Then on **host** (HF token):

```bash
OAT_GIT_COMMIT=... GPU=0 bash scripts/cluster_latency_awr16_hf.sh
```

Script skips suites that already have `modes.awr16` in `latency_fair_kv_n16.json`.

---

## 5. Acceptance (per suite)

Open `output/eval/matched_s10000/robocasa/<task>/latency_fair_kv_n16.json`:

- [ ] `"fair_kv": true`
- [ ] `modes.awr16` present (Single/BoN16/BoN32 already there — do not wipe)
- [ ] `protocol.n_trials == 8`, `reps == 10`, `warmup == 20`, `batch_size == 1`
- [ ] `awr16` uses **`predict_action (KV)`**
- [ ] Paper cell = `trial_median_mean_ms` ± `trial_median_std_ms`
- [ ] Locked `latency_fair_kv.json` **unchanged**

Quick check:

```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path("output/eval/matched_s10000/robocasa/close_drawer/latency_fair_kv_n16.json")  # CHANGE
m = json.loads(p.read_text())["modes"]["awr16"]
print(f"{m['trial_median_mean_ms']:.1f}±{m['trial_median_std_ms']:.1f}")
PY
```

---

## 6. Fill RESULTS.md Table C′

For each finished RC task, set **AWR16 (KV)** to `**XX.X±Y.Y**` (1 decimal) in:

- `close_drawer`
- `turn_off_sink_faucet`
- `turn_off_microwave`

Do **not** change Single / BoN8 / BoN16 / BoN32 columns for other suites.

---

## 7. Done criteria

1. All **3** RC suites have `modes.awr16` in `latency_fair_kv_n16.json`.  
2. No orphan ~GB ckpts on disk.  
3. `RESULTS.md` Table C′ AWR16 filled for those three.  
4. Report: `suite  XX.X±Y.Y ms  git=<commit>  gpu=V100`.

---

## 8. Files

| File | Role |
|------|------|
| `scripts/measure_latency_paper.py` | `--fair_kv --awr16_ckpt …` |
| `scripts/cluster_latency_awr16_hf.sh` | disk-safe HF cycle (extend `ALL_JOBS` for RC only) |
| `AGENT_GUIDE_TABLE_C_PRIME_LATENCY.md` | full C′ protocol |
| `RESULTS.md` § Table C′ | numbers to fill |
| HF [`Mirageinv/AWR`](https://huggingface.co/Mirageinv/AWR) | Bon16-AWR@100ep ckpts (upload when ready) |
