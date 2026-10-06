---
license: mit
library_name: other
tags:
  - robotics
  - oat
  - imitation-learning
  - robomimic
  - metaworld
  - robocasa
  - benchmark
  - latency
pretty_name: AAAI OAT Paper Models + Eval Artifacts
---

# AAAI OAT — canonical artifact pack (models + eval)

**This repo is the primary copy for paper artifacts.**  
Training data lives separately in [`hackhackhack66666/aaai-datasets`](https://huggingface.co/datasets/hackhackhack66666/aaai-datasets).

Paper track: **Consensus Selection (CS-D) on OAT** — matched baseline / BoN / AWR across RoboMimic, MetaWorld, RoboCasa.

---

## Start here (reviewer navigation)

| If you need… | Open this first |
|--------------|-----------------|
| Final paper numbers + protocol | `docs/RESULTS.md` |
| **Selector baselines on top of PDF (AAMAS27)** | **`docs/aamas27/README.md`** (this pack) + `eval/aamas27_selector/` |
| RoboCasa-specific protocol / literal-5 seeds | `docs/RESULTS_ROBOCASA.md` |
| Machine-readable file list | `INVENTORY.txt` |
| Per-suite SR summaries | `eval/matched_s10000/<suite>/summary.json` or `.../robocasa/<task>/summary_literal5.json` |
| Latency tables | `eval/matched_s10000/table_c.json`, `table_c_fair_kv.json` |
| Checkpoints to rerun eval | `checkpoints/my_models/` + `checkpoints/selected_from_output/` + `checkpoints/all_training/` |
| Train configs | `hydra/<run>/.hydra/config.yaml` |
| Raw training data | **datasets repo** (not here) |

---

## Two-repo layout

| Repo | Role | Size (approx) |
|------|------|---------------|
| **`hackhackhack66666/aaai-datasets`** | Zarr + retained HDF5 for training | ~22 GB |
| **`hackhackhack66666/aaai27-models`** (this repo) | Checkpoints, matched eval, latency, logs, Hydra, docs | ~50+ GB |

---

## Directory map

```text
README.md                         # this file
MANIFEST.json                     # pack metadata
INVENTORY.txt                     # flat file listing (generated at upload)

docs/
  RESULTS.md                      # paper ledger (Table P, latency, manifests)
  RESULTS_ROBOCASA.md
  ROBOCASA.md / ROBOMIMIC.md / METAWORLD*.md
  AGENT_GUIDE_TABLE_C_PRIME_LATENCY.md

checkpoints/
  my_models/                      # Wave2 AWR + RoboCasa TopK locks (+ lock txt)
  selected_from_output/           # paper-primary tokenizer/base ckpts (subset)
  all_training/                   # ALL retained TopK/latest ckpts from output runs
  awr16_from_hf/                  # BoN16-distill ckpts mirrored from Mirageinv/AWR

hydra/
  output/<run>/.hydra/            # train configs for paper runs

eval/
  matched_s10000/                 # Table P source-of-truth eval outputs
  eval_out/                       # replan probes + RoboCasa obs audit artifacts
  aamas27_selector/               # CS vs KDPE / selector baselines (on top of PDF)
    aic4/                         # dump from lab-c-0 (only cluster with these runs)
      paired/                     # [E] paired vote/kdpe (+ LIBERO 6 methods)
      paired_seed/                # [C] --policy_seed 0
      aamas27_selector_baselines/ # vote/random/medoid/max_ll (+ RC)
      kdpe_n8/                    # earlier KDPE cells + episodes.jsonl
      latency/                    # Table 5 selector + full-forward JSON/MD/logs
      queues/                     # cell_*.sh / lane_*.sh launchers (reproduce evals)
    ccm/                          # dump from ccmplanner (V100 cells; RC discarded from paper)
      aamas27_selector_baselines/ # close-drawer vote/random/medoid json live here
      ccm_v100_paired/            # can/lift base+vote V100 paired
    INDEX.txt                     # every eval_log.json + latency json
    MANIFEST_AAMAS27.json

docs/aamas27/
  LIBERO_LONG_REZULTATY.md        # LIBERO SR + latency (Russian ledger)
  TABLE5_LATENCY.md
  CS_VS_KDPE_TABLE.md
  SELECTOR_BASELINES_STATUS.md
  AAMAS27_DUMP.md
  SELECTOR_BASELINES_PROTOCOL.md


logs/
  matched_s10000_*.log
  awr_s10000_*wave2*.log
  mw_*bon16_32*.log
  replan_probe_*.log
```

---

## Table P — where the numbers live

**Protocol (RM/MW):** `test_start_seed=10000`, `n_test=50`, `-n 5`, OAT8, BoN `--bon_free 8 --bon_signal vote`.

**Protocol (RoboCasa):** literal 5 seeds `10000..10004`, each `-n 1 --n_test 50`.

### RoboMimic + MetaWorld (matched_s10000 layout)

| Suite | Summary | Baseline | BoN8 | AWR | Extra |
|-------|---------|----------|------|-----|-------|
| can | `can/summary.json` | `can/baseline_n5/eval_log.json` | `can/bon_n8_n5/` | `can/awr_n5/` | `can/latency*.json` |
| lift (run A, ep900) | `lift/summary.json` | `lift/baseline_n5/` | `lift/bon_n8_n5/` | `lift/awr_n5/` | — |
| lift (run B, ep1400) | `lift_ep1400/summary.json` | `lift_ep1400/baseline_n5/` | `lift_ep1400/bon_n8_n5/` | `lift_ep1400/awr_n5/` | paper primary Lift |
| square | `square/summary.json` | `square/baseline_n5/` | `square/bon_n8_n5/` | `square/awr_n5/` | replan probe in `eval/eval_out/` |
| coffee-pull | `coffee-pull/summary.json` | … | … | … | BoN16/32: `bon_n16_n5/`, `bon_n32_n5/` |
| stick-pull | `stick-pull/summary.json` | … | … | … | BoN16/32 |
| disassemble | `disassemble/summary.json` | … | … | … | BoN16/32 |
| box-close | `box-close/summary.json` | … | … | … | BoN16/32 |

All paths under `eval/matched_s10000/`.

### RoboCasa (literal-5 layout)

| Task | Summary | Per-seed evals |
|------|---------|----------------|
| close_drawer | `robocasa/close_drawer/summary_literal5.json` | `baseline_seed1000X/`, `bon_n8_seed1000X/` |
| coffee_press_button | `robocasa/coffee_press_button/summary_literal5.json` | same pattern |
| turn_off_sink_faucet | `robocasa/turn_off_sink_faucet/summary_literal5.json` | same |
| turn_off_microwave | `robocasa/turn_off_microwave/summary_literal5.json` | same |

Latency (AWR16/KV extension): `robocasa/<task>/latency_fair_kv_n16.json`.

---

## Checkpoints — paper-primary paths

### Wave2 AWR (deployable distill)

| Suite | Checkpoint |
|-------|------------|
| can | `checkpoints/my_models/awr_s10000_can.ckpt` |
| lift (run A) | `checkpoints/my_models/awr_s10000_lift.ckpt` |
| square | `checkpoints/my_models/awr_s10000_square.ckpt` |
| coffee-pull | `checkpoints/my_models/awr_s10000_coffee-pull.ckpt` |
| stick-pull | `checkpoints/my_models/awr_s10000_stick-pull.ckpt` |
| disassemble | `checkpoints/my_models/awr_s10000_disassemble.ckpt` |
| box-close | `checkpoints/my_models/awr_s10000_box-close.ckpt` |

### RoboCasa TopK locks (Wave1 base)

| Task | Checkpoint | Lock file |
|------|------------|-----------|
| close_drawer | `robocasa_close_drawer_topk_ep0500_sr0.700.ckpt` | `robocasa_close_drawer_topk_lock.txt` |
| coffee_press_button | `robocasa_coffee_press_button_topk_ep0500_sr0.600.ckpt` | `…_topk_lock.txt` |
| turn_off_sink_faucet | `robocasa_turn_off_sink_faucet_topk_ep0500_sr0.580.ckpt` | `…_topk_lock.txt` |
| turn_off_microwave | `robocasa_turn_off_microwave_topk_ep0500_sr0.620.ckpt` | `…_topk_lock.txt` |

All under `checkpoints/my_models/`.

### Base + tokenizer (selected paper ckpts)

Under `checkpoints/selected_from_output/output/<run>/checkpoints/`.  
See `docs/RESULTS.md` § *Per-suite train → paper base* for the canonical mapping.

### Full training TopK archive

**All retained** training checkpoints (TopK + `latest.ckpt`) are under:

`checkpoints/all_training/home/askhabaliev_gs/mipt_paper/oat/output/<date>_<run>/checkpoints/*.ckpt`

> Note: paths preserve the original cluster absolute prefix from upload staging.  
> Paper-primary subset is also mirrored under the cleaner prefix:  
> `checkpoints/selected_from_output/output/<date>_<run>/checkpoints/*.ckpt`

Use `all_training/` if you need non-primary epochs (e.g. Lift ep-0900 vs ep-1400, Square TopK trajectory).

### BoN16-distill (latency / extended)

Mirrored under `checkpoints/awr16_from_hf/`:

- `robomimic_{can,lift,square}_awr_bon16_e100.ckpt`
- `robocasa_coffee_press_button_awr_bon16_e100.ckpt`

---

## Latency (Table C / C')

| Table | Aggregate | Per-suite |
|-------|-----------|-----------|
| **C** (deployed path) | `eval/matched_s10000/table_c.json` | `<suite>/latency.json` |
| **C′** (fair-KV) | `eval/matched_s10000/table_c_fair_kv.json` | `<suite>/latency_fair_kv.json` |
| **C′ AWR16/KV ext** | — | `robocasa/<task>/latency_fair_kv_n16.json` (+ RM/MW `*_n16.json`) |

Protocol details: `docs/AGENT_GUIDE_TABLE_C_PRIME_LATENCY.md`.

---

## AAMAS27 — selector baselines **on top of the PDF** (CS / KDPE / random / medoid / max_ll)

These are **not** Table P BoN-vs-base. They are the extra selector comparison (Consensus Selection vs KDPE and ablations) plus Table 5 H100 selector latency.

**Clusters scanned:** aicenter4/`lab-c-0` (**paper host**, 102 `eval_log.json` + latency), ccmplanner (**audit dump**, 28 `eval_log.json` including close-drawer vote/random/medoid), aicenter1 (**empty**), cds2 (**empty**). aicenter3 not used.

Git branch: `aamas27_selector_baselines`  
Code eval: cluster `~/oat_code_kdpe`  
Latency: cluster `~/oat_code_seed` (`LAST_CALL` in `kdpe.py`)  
Do **not** mix **[E]** (`eval/aamas27_selector/aic4/paired/`, no `--policy_seed`) with **[C]** (`…/paired_seed/`, `--policy_seed 0`).

### Two clocks (latency)

| Clock | What | Typical N=8 D=7 | Paper table |
|-------|------|-----------------|-------------|
| **Selector-only** | `_bon_select` / KDPE on tensor `[N,R,D]` | CS **0.264–0.272 ms**, KDPE **0.53–0.55 ms** (~2×) | **Table 5** |
| **Full forward** | vision CNN + AR sample N + detokenize + select | CS/KDPE both **~28 ms** (Δ ~0.3 ms, ~1%) | context only |

Do not write “CS speeds up inference 2×”. That 2× is selector-only. LIBERO D=7 matches RoboMimic can. MetaWorld D=4: CS still ~0.27 ms, **KDPE ~0.20 ms — do not pool with D=7**.

### Where the numbers live

Root on Hub: `eval/aamas27_selector/aic4/`  
Cluster original: `~/oat_eval_out/` (same relative paths).

| Need | Path under `eval/aamas27_selector/` |
|------|--------------------------------------|
| LIBERO-LONG [E] suite + per-task | `aic4/paired/libero10/{vote,medoid,kdpe,max_likelihood,base,random}/eval_log.json` |
| RM/MW [E] CS vs KDPE (paired) | `aic4/paired/<task>/{vote,kdpe}/eval_log.json` (+ `base/` on can/lift/square) |
| RM/MW [E] random/medoid/max_ll | `aic4/aamas27_selector_baselines/<task>/{vote,random,medoid,max_likelihood}_n8/eval_log.json` |
| RoboCasa [E] (aic4 trusted) | `aic4/aamas27_selector_baselines/{close-drawer,coffee-button,faucet-off,microwave-off}/` |
| Confirmatory [C] | `aic4/paired_seed/<task>/{vote,kdpe}/eval_log.json` |
| Episode-level jsonl (paired) | `aic4/paired/<task>/<method>/episodes.jsonl` |
| Table 5 canon (can, selector+gen) | `aic4/latency/selector_latency_gpu1_n32diag.json` |
| can full-forward replicate | `aic4/latency/selector_latency_gpu1_clean_20261006.json` |
| **LIBERO latency** (selector+gen+full) | `aic4/latency/selector_latency_gpu1_libero10.json` |
| lift/square + MW D=4 latency | `aic4/latency/selector_latency_gpu2_{lift,square,box-close,coffee-pull,stick-pull,disassemble}.json` |
| Human ledgers | `docs/aamas27/*.md` |
| Launch scripts (aic4) | `aic4/queues/` (`cell_*.sh`, `lane_*.sh`) |
| ccm close-drawer vote/random/medoid json | `ccm/aamas27_selector_baselines/close-drawer/{vote,random,medoid}_n8/eval_log.json` |
| ccm V100 paired can/lift | `ccm/ccm_v100_paired/{can,lift}/{base,vote}/eval_log.json` |

Paper scalars:

| Suite | File | Key |
|-------|------|-----|
| LIBERO | `paired/libero10/<method>/eval_log.json` | `mean_success_rate_mean` (± `…_std`, n=5) |
| RM/MW/RC | same `eval_log.json` | `mean_success_rate_mean` (reported as % in git docs) |

LIBERO [E] (n_test=500, seed 10000, N=8): vote **0.665±0.025** · medoid **0.663±0.028** · kdpe **0.605±0.009** · max_ll **0.584±0.010** · base **0.558±0.018** · random **0.554±0.015**.

LIBERO latency N=8 empty GPU1: selector CS **0.272** / KDPE **0.553**; full **27.76 / 28.09**; gen **22.73**.

### Discarded / do not use as paper

- **ccm RoboCasa coffee/microwave/faucet** — dumped under `ccm/` for audit, **not paper** (systematically low vs aic4).
- `aic4/latency/selector_latency_gpu0.json` and `gpu1.json` (warmup-10 / occupied-GPU drafts). Canon = `n32diag`. Full can = `*_clean_20261006`.
- `aic4/paired/square/{vote2,kdpe2}` extra repeats — keep for audit; primary square cells are `vote`/`kdpe`.
- close-drawer vote/medoid: **json on ccm**, aic4 dirs may lack `eval_log.json`. Paper close-drawer vote **56.8±5.6** is the aic4-trusted number in STATUS (not the discarded ccm RC SR).

### Download this pack only

```bash
huggingface-cli download hackhackhack66666/aaai27-models \
  --include "eval/aamas27_selector/**" \
  --include "docs/aamas27/**" \
  --local-dir ./aaai27-models
```

---

## Known missing artifacts (documented honestly)

These were **never saved** locally during the latency workflow (`download → measure → delete`):

| Missing file | Impact |
|--------------|--------|
| `robocasa_close_drawer_awr_bon16_e100.ckpt` | cannot reload AWR16 weights; **latency JSON exists** |
| `robocasa_turn_off_sink_faucet_awr_bon16_e100.ckpt` | same |
| `robocasa_turn_off_microwave_awr_bon16_e100.ckpt` | same |
| `awr_s10000_lift_ep1400.ckpt` | Lift run B AWR weights not in `my_models/`; **eval_log + summary exist** |

**Not blocking for Table P SR reproduction** (baseline/BoN/AWR Wave2 primary ckpts are present).  
Blocks only exact re-run of those specific AWR16 latency measurements from weights.

---

## How to download

```bash
pip install -U huggingface-hub

# full repo (large)
huggingface-cli download hackhackhack66666/aaai27-models --local-dir ./aaai27-models

# one suite eval only
huggingface-cli download hackhackhack66666/aaai27-models \
  eval/matched_s10000/can/summary.json \
  --local-dir ./aaai27-models

# one checkpoint
huggingface-cli download hackhackhack66666/aaai27-models \
  checkpoints/my_models/awr_s10000_can.ckpt \
  --local-dir ./aaai27-models
```

Datasets (separate):

```bash
huggingface-cli download hackhackhack66666/aaai-datasets --repo-type dataset --local-dir ./aaai-datasets
```

---

## Related

- Datasets: https://huggingface.co/datasets/hackhackhack66666/aaai-datasets
- Upstream OAT: https://github.com/Chaoqi-LIU/oat
- External AWR16 mirror (partial): https://huggingface.co/Mirageinv/AWR

---

## Changelog

| Date | Note |
|------|------|
| 2026-08-19 | Initial upload: matched eval, my_models, selected ckpts, latency, logs, hydra, docs |
| 2026-08-19 | Added `checkpoints/all_training/` (full TopK archive), expanded README navigation |
| 2026-10-06 | Added `eval/aamas27_selector/` + `docs/aamas27/`: all aic4 selector-baseline evals, LIBERO-LONG, Table 5 + full-forward latency |
