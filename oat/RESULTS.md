# OAT Reproduction — Results & Artifacts

Cluster container: `oat_mipt_robomimic_askhabaliev_gs`  
Workspace root: `/workspace/oat` (paths below are relative to `oat/`)

**Eval protocol notes:**
- **Chain5 (internal 250-episode estimate):** one trained checkpoint, 5 disjoint environment-seed blocks × 50 rollouts; `summary.json` in eval dir. This is **not** the paper's 5 independently trained seeds.
- **Train-time policy eval:** RoboMimic uses `n_test=50`, normally every 100 epochs; MT4 multitask used `n_test=50`, every 50 epochs; MetaWorld single-task specialists use `n_test=250`, every 200 epochs. Checkpoints are top-k by `mean_success_rate`.
- **BoN / AWR (quick pipeline):** `--n_test 50`, `--num_exp 3`, BoN `N=8 vote`; `eval_log.json` in eval dir.
- All use **OAT8:** `--use_k_tokens 8 --entropy_threshold 0`, `MUJOCO_GL=egl`.

**Cluster snapshot (2026-07-15 ~01:05 MSK):** RoboMimic + MT4 exploratory **DONE**. coffee/stick **BoN+AWR DONE**. disassemble chain5 **DONE**. box-close train plateaued (best **ep-2000 @ 55.2%**); **matched baseline RUNNING** (`mwst_matched_box_close`). Paper matched rematch (baseline −n 3 for all suites) still mostly TBD — see matched table below.

**Status overview**

| Block | Status |
|-------|--------|
| RoboMimic (Lift / Can / Square) | **DONE** exploratory; **matched baseline rematch TBD** |
| MetaWorld MT4 multitask | **DONE** exploratory; **matched baseline rematch TBD** |
| MetaWorld single-task data + tokenizer | **DONE** |
| MetaWorld single-task policy train | **DONE** — all four specialists stopped |
| MetaWorld single-task chain5 | coffee / stick / disassemble **DONE**; box-close **skip for now** |
| MetaWorld single-task BoN / AWR | coffee / stick **DONE**; box / disassemble **TBD** |
| Matched (ICRA) | box-close baseline **RUNNING**; others TBD (see Table B) |

---

## Summary table

| Benchmark | Chain5 baseline (sanity) | BoN N=8 (3 exp) | AWR single (3 exp) | Δ_BoN vs chain5† | Δ_AWR vs chain5† |
|-----------|--------------------------|-----------------|--------------------|------------------|------------------|
| **Lift** | **83.6 ± 1.7%** (ep-0600) | 91.3 ± 2.9% | **93.3 ± 1.8%** | +7.7 pp | +9.7 pp |
| **Can** | **86.0 ± 2.8%** (ep-1700) | 90.7 ± 1.3% | 89.3 ± 1.8% | +4.7 pp | +3.3 pp |
| **Square** | *(see Table P matched)* | matched BoN **28.0±9.4%** | matched AWR **31.2±5.2%** | matched Δ **−8.0** | matched Δ **−4.8** |
| **MT4 multitask** ‡ | **28.4 ± 3.1%** (ep-0450) | 26.7 ± 2.4% | 18.7 ± 1.8% | −1.7 pp | −9.7 pp |
| **MW coffee-pull** (specialist) | rerun pending | — | — | — | — |
| **MW stick-pull** (specialist) | **16.4 ± 4.0%** (ep-0800) | **30.7 ± 3.1%** | **39.3 ± 7.6%** | exploratory +14.3 pp | exploratory +22.9 pp |
| **MW disassemble** (specialist) | **66.4 ± 3.2%** (ep-1400) | TBD | TBD | — | — |
| **MW box-close** (specialist) | *train best* **55.2%** @ep-2000 (chain5 N/A) | TBD | TBD | — | — |

**Square note:** matched Table P **DONE** (base 36.0 / BoN 28.0 / AWR 31.2 @ `matched_s10000/square/`). BoN hurts; highest replan (probe **20.4**). Old pre-2026-07-20 exploratory artifacts deleted — do not cite.

† **Exploratory only — not the paper matched estimator.** Chain5 = 5×50 distinct inits; quick BoN/AWR = 3 stochastic runs on seeds `1000–1049`. Paper Δ requires matched baseline/BoN/AWR on one **shared fixed init set** (`test_start_seed=1000`, `n_test=50`, `-n 3`). See [`RESOLUTIONPLAN.md`](RESOLUTIONPLAN.md).

‡ **MT4:** kept for the exploratory record; **not** in matched Table B / paper (BoN/AWR did not help).

Paper targets (Table VI, OAT₈): Lift **99.2%**, Can **80.8%**, Square **39.2%**, MT4 per-task specialists **44.4 / 26.4 / 17.2 / 9.6%**, MT4 avg **24.4%**.

---

## Publication-readiness audit (ICRA)

**This repo track (us):** RoboMimic + MetaWorld — matched BoN/AWR on fixed OAT checkpoints.  
**LIBERO / adaptive K–R diagnosis / breadth (other suites, DP, contact-rich):** scientific lead — referenced as story context and protocol example, not our build scope.  
**Intended RM/MW claim:** relative Δ of BoN/AWR vs a **matched** single-sample baseline on the same shared fixed init set. Table VI absolute parity is a sanity check only.

### Verified facts

- Completed exploratory SR values match cluster `summary.json` / `eval_log.json`.
- Within each pipeline, BoN and AWR share the same base checkpoint; AWR is distilled from BoN rollouts of that checkpoint.
- AWR collection uses env seeds near 0 (worker offsets); final eval defaults to seed base 1000 — no direct collect/eval seed overlap observed on completed RoboMimic sets.
- RoboMimic = official multi-human image Zarr; MetaWorld single-task = locally regenerated demos (port caveats below).

### Limitations

- **Exploratory vs paper claim (RM/MW).** Summary Δ vs **chain5** are **exploratory** until matched rematch. Paper main text for this track uses only matched Δ (baseline, BoN, AWR on one shared fixed init set, `-n 3`). Chain5 absolute SR = sanity/appendix, not the BoN comparator.
- **Shared fixed init set (not “held-out”).** Matched protocol uses `test_start_seed=1000`, `n_test=50` (episodes `1000–1049`). This is the **canonical eval pool** and often overlaps train-time checkpoint selection — do **not** call it an unused held-out test set unless a new seed base is chosen. Prefer the term **matched-seed estimator**, not classic per-episode paired testing.
- **Comparison protocol.** Chain5 = five disjoint seed blocks × 50 (`num_exp=1`); exploratory BoN/AWR = three stochastic repeats on one block. Uncertainty and Δ are not comparable; sign/magnitude can flip (Can, MT4, coffee).
- **n_test / suite mismatch.** LIBERO lead track typically uses `n_test=500`; our RM/MW matched table uses `n_test=50`. Do not pool suites into one causal claim without an explicit protocol note.
- **Can / coffee / stick / disassemble / box:** same matched rematch; no suite pre-labeled from exploratory runs. Paper includes a suite only after Table B cells are filled from `matched/`.
- **Lift:** matched baseline logged, но paper path = **retrain** → new matched triplet later.
- **Square:** matched Wave1+2 **DONE**; BoN negative (Δ−8.0), highest replan — failure-mode suite for vote.
- **MT4 multitask.** Exploratory only; **not** in matched/paper wave.
- **Fixed-policy vs Table VI.** Single OAT checkpoint under BoN/AWR vs paper’s multi-seed Table VI — absolute parity is sanity only.
- **Checkpoint selection.** Some ckpts selected on the same eval pool; matched Δ on a fixed ckpt is the claim.
- **Meta-World port.** Controlled demo port differs from original sim-env success criterion — interpret MW within this implementation.
- **Compute cost.** BoN costs N candidates per replan; report latency/cost with SR (Table C, all suites).
- **Claim scope.** LIBERO = lead diagnosis; RM/MW = matched generalization probes.

> **PAPER = Table P only** (`matched_s10000`, episodes `10000–10049`).  
> Table B below = lab draft on selection seeds `1000–1049` — **не в статью**. Exploratory Summary = тоже только lab.

### Table P — PAPER matched (seed 10000) ← сюда пишем числа для статьи

**Protocol:** `test_start_seed=10000`, `n_test=50`, `-n 5`, OAT8, `--temperature 1.0 --topk 10`, BoN `--bon_free N --bon_signal vote` (primary **N=8**; MW N-sweep also **N=16/32**).  
**Δ** = method − paper baseline. Ckpt выбирался на seed 1000; отчёт на 10000 (disjoint).  
**Wave 1:** baseline + BoN N=8. **Wave 1b (MW only, locked):** BoN N=16 + N=32 on same ckpt/seeds.  
**Wave 2:** свежий AWR (не exploratory).  
**Launch:** `bash scripts/cluster_launch_matched_paper_wave.sh` · MW BoN16/32: `logs/mw_*_bon16_32_*.log` / `mw_*_bon32_only_*.log`.

#### ⛔ GATE → Wave 2 (зафиксировано 2026-07-16)

Пока Gate не закрыт — **AWR не запускать**. Полный текст: [`RESOLUTIONPLAN.md`](RESOLUTIONPLAN.md).

1. Wave 1 suite DONE: есть `summary.json` + baseline/BoN в Table P (**только** `matched_s10000`).
2. ❌ Не брать: exploratory `policy_awr_*.ckpt` / старые `awr_*.npz` / seed-1000 Table B / chain5 как paper Δ.
3. Collect: явный `--seed 0` (❌ selection RM `1000–1049` / MW `1000–1249`, ❌ report `10000`); eval AWR: `TEST_START_SEED=10000`.
4. Square: first **ep-1500**; если BoN flat → rematch **ep-0600** @ s10000 (exploratory 600 ≠ paper).
5. Оригинал OAT Table VI — **sanity only**, не comparator.
6. **RM vs MW фиты разные** (`cfg.seed` 42 vs 0), но env TopK у обоих default `test_start_seed=1000` (MW длиннее до 1249). Paper `10000` вне обоих.

**После anti-leak seeds:** остаётся (a) **MW demo port** = controlled limitation; (b) **Lift run B** AWR eval + coffee/square Wave2; (c) RoboCasa policies serial. MT4 — не в paper.

| Suite | Base ckpt | Paper baseline | BoN N=8 | AWR | Δ_BoN | Δ_AWR | Artifacts |
|-------|-----------|----------------|---------|-----|-------|-------|-----------|
| Can | ep-1700 | **76.4±3.0%** | **80.8±1.1%** | **79.2±7.7%** | **+4.4** | **+2.8** | `matched_s10000/can/` · `awr_s10000_can.*` (AWR < BoN) |
| MW coffee-pull | **ep-1000** TopK lock `090816` | **40.8±2.3%** | **43.2±4.8%** | **41.2±2.3%** | **+2.4** | **+0.4** | Wave1+2 **DONE** · N-sweep below · `matched_s10000/coffee-pull/` · `awr_s10000_coffee-pull.*` · replan **12.3** |
| MW stick-pull | ep-0800 | **15.6±6.2%** | **25.6±2.6%** | **26.8±5.2%** | **+10.0** | **+11.2** | N-sweep below · `matched_s10000/stick-pull/` · `awr_s10000_stick-pull.*` |
| MW disassemble | ep-1400 | **62.4±5.2%** | **63.2±6.3%** | **69.6±5.7%** | **+0.8** | **+7.2** | N-sweep below · `matched_s10000/disassemble/` · `awr_s10000_disassemble.*` |
| MW box-close | ep-2000 | **59.6±7.5%** | **66.4±3.0%** | **72.8±4.1%** | **+6.8** | **+13.2** | N-sweep below · `matched_s10000/box-close/` · `awr_s10000_box-close.*` |
| Square | **ep-0700** TopK lock `215024` | **36.0±7.5%** | **28.0±9.4%** | **31.2±5.2%** | **−8.0** | **−4.8** | Wave1+2 **DONE** · BoN **worst** Δ · AWR partial recover · replan probe **20.4** (highest) · `matched_s10000/square/` · `awr_s10000_square.*` · `eval_out/replan_probe_square/` |
| Lift | **2 runs** (see below) | | | | | | |
| — run A (mid-train) | **ep-0900** `144024` | **82.0±3.2%** | **84.8±4.6%** | **81.2±4.1%** | **+2.8** | **−0.8** | `matched_s10000/lift/` · `awr_s10000_lift.*` · replan **8.0** |
| — run B (TopK lock / paper) | **ep-1400** `144024` | **87.6±2.2%** | **84.0±4.9%** | **85.2±3.9%** | **−3.6** | **−2.4** | `matched_s10000/lift_ep1400/` · `awr_s10000_lift_ep1400.*` · BoN&AWR < base · replan **9.3** |
| **RoboCasa** *(literal-5 mean±SEM; see § below)* | | | | | | | |
| close_drawer | **ep-0500 @0.700** lock | **56.0±1.1%** | **59.6±1.6%** | — | **+3.6±1.9** | — | Wave1 **DONE** · `summary_literal5.json` · `my_models/robocasa_close_drawer_topk_ep0500_sr0.700.ckpt` |
| coffee_press_button | **ep-0500 @0.600** lock | **44.0±3.0%** | **55.6±2.7%** | — | **+11.6±4.1** | — | Wave1 **DONE** · `summary_literal5.json` · `my_models/robocasa_coffee_press_button_topk_ep0500_sr0.600.ckpt` |
| turn_off_sink_faucet | **ep-0500 @0.580** lock | **52.4±2.5%** | **56.0±3.4%** | — | **+3.6±4.2** | — | Wave1 **DONE** · `summary_literal5.json` · `my_models/robocasa_turn_off_sink_faucet_topk_ep0500_sr0.580.ckpt` |
| turn_off_microwave | **ep-0500 @0.620** lock | **43.6±3.1%** *(base 5/5)* | Wave1 BoN **RUNNING** (1/5) | — | — | — | `my_models/robocasa_turn_off_microwave_topk_ep0500_sr0.620.ckpt` |

#### Table P — MW BoN N-sweep (N=8/16/32) ← **LOCKED 2026-07-26**

Same paper protocol as Table P (`test_start_seed=10000`, `n_test=50`, `-n 5`, OAT8, `vote`).  
**Source of truth (numbers):** `output/eval/matched_s10000/<suite>/bon_n{8,16,32}_n5/eval_log.json` → keys `mean_success_rate_mean` ± `mean_success_rate_std` (`num_exp=5`).  
Primary paper BoN column above remains **N=8**; this block = inference-scaling appendix / figure. Paths relative to `/workspace/oat`.

| Suite | baseline | BoN N=8 | BoN N=16 | BoN N=32 | Δ₁₆ | Δ₃₂ |
|-------|----------|---------|----------|----------|-----|------|
| stick-pull | **15.6±6.2%** | **25.6±2.6%** | **28.4±3.8%** | **29.6±8.8%** | **+12.8** | **+14.0** |
| coffee-pull | **40.8±2.3%** | **43.2±4.8%** | **43.2±3.3%** | **41.2±3.0%** | **+2.4** | **+0.4** |
| disassemble | **62.4±5.2%** | **63.2±6.3%** | **64.8±5.8%** | **64.8±3.0%** | **+2.4** | **+2.4** |
| box-close | **59.6±7.5%** | **66.4±3.0%** | **70.4±3.6%** | **70.8±4.1%** | **+10.8** | **+11.2** |

**Read (locked):** N=16→32 **flat / saturating** on disassemble & box-close (N32≈N16); coffee **regresses** at N=32 vs N=8/16; stick edges up but N=32 std blows up (8.8). Paper BoN column stays **N=8**; N=16 = useful knee (box-close +10.8, stick +12.8).

##### Reproduce manifest — BASE_CKPT + eval dirs + logs + launchers

| Suite | BASE_CKPT (same as Table P Wave1) | eval dirs (SR) | run logs | launcher |
|-------|-----------------------------------|----------------|----------|----------|
| stick-pull | `output/20260711/134439_train_oatpolicy_mw-stick-pull_st_N50/checkpoints/ep-0800_sr-0.212.ckpt` | `output/eval/matched_s10000/stick-pull/{bon_n16_n5,bon_n32_n5}/eval_log.json` | `logs/mw_stick_bon16_32_20260724_221255.log` | `scripts/_run_mw_tablep_bon16_32.sh stick-pull <gpu> <ckpt>` |
| coffee-pull | `output/20260720/090816_train_oatpolicy_mw-coffee-pull_st_N50/checkpoints/ep-1000_sr-0.432.ckpt` | `output/eval/matched_s10000/coffee-pull/{bon_n16_n5,bon_n32_n5}/eval_log.json` | `logs/mw_coffee_bon16_32_20260724_221421.log` | idem |
| disassemble | `output/20260711/134440_train_oatpolicy_mw-disassemble_st_N50/checkpoints/ep-1400_sr-0.700.ckpt` | `output/eval/matched_s10000/disassemble/{bon_n16_n5,bon_n32_n5}/eval_log.json` | `logs/mw_disassemble_bon16_32_20260725_061513.log` · **N32 rerun** `logs/mw_disassemble_bon32_only_20260725_161434.log` | `_run_mw_tablep_bon16_32.sh` then `_run_mw_tablep_bon32_only.sh` (canonical N32 = `*_bon32_only_*`) |
| box-close | `output/20260711/134439_train_oatpolicy_mw-box-close_st_N50/checkpoints/ep-2000_sr-0.552.ckpt` | `output/eval/matched_s10000/box-close/{bon_n16_n5,bon_n32_n5}/eval_log.json` | `logs/mw_box-close_bon16_32_20260725_061513.log` · **N32 rerun** `logs/mw_box-close_bon32_only_20260725_175116.log` | idem |

**Also:** queue orchestrator `scripts/_queue_mw_bon16_32_wave2.sh` · log `logs/mw_bon16_32_queue_20260724_223906.log`.  
**Flags baked in launchers:** `-n 5 --n_test 50 --test_start_seed 10000 --use_k_tokens 8 --entropy_threshold 0 --temperature 1.0 --topk 10 --bon_free {16\|32} --bon_signal vote --n_parallel_envs 2`.  
**Verified on cluster 2026-07-26:** all 8× `bon_n{16,32}_n5/eval_log.json` present; ckpts exist; listed logs exist.  
**Note:** `summary.json` per suite still encodes Wave1 N=8 (+AWR) only — do **not** expect N16/32 inside it; cite `eval_log.json` above.
**Wave1 BoN verified (2026-07-16):** can / stick — `eval_log.json` ↔ `summary.json` match. Primary artifacts = eval_logs under `matched_s10000/`.

**Status (2026-07-28 ~13:20 MSK):** MW Table P BoN N-sweep **LOCKED**. RM/MW Wave1+2 **DONE**. **RoboCasa Wave1:** coffee + close + **sink DONE** (literal-5); microwave base 5/5 / BoN running (1/5). **Table C′:** RC Single+BoN{8,16,32} **DONE**; RM/MW BoN16/32 **DONE**; **AWR16 DONE** for HF set (can/lift/square/coffee); remaining **RoboCasa only** AWR16 = `close_drawer` / `turn_off_sink_faucet` / `turn_off_microwave` (no ckpt on HF yet — see `AGENT_GUIDE_AWR16_LATENCY_REMAINING.md`).

**Square read (high replan × worst BoN):** see replan § below — long horizon + weak base → many replans → vote compounds; AWR only half-recovers.

**Cluster ops (2026-07-23):** deleted junk RC policy dirs (0 TopK, dead): `20260721/202439_*coffee*` (empty hydra), `20260723/041317_*microwave*` (OOM), `20260723/041317_*sink*` (killed mid ep0). **Kept:** coffee `204916` (TopK), close `041317` (live), all 4 tokenizers.

**Coffee-pull TopK lock (2026-07-21T23:00Z):** train-eval after ep-1000 did **not** beat 0.432 → lock `ep-1000_sr-0.432.ckpt`. Wave1 **DONE** (base 40.8 / BoN 43.2, Δ=+2.4).

**Square TopK lock (2026-07-22T10:11Z):** plateau rule peak-then-2-below → `ep-0700_sr-0.420.ckpt`; `my_models/square_topk_lock.txt`; train killed; Wave1 started.

**Artifact pattern list:**

| Artifact pattern | Examples |
|------------------|----------|
| AWR dataset | `my_datasets/awr_s10000_{can,stick-pull,box-close,disassemble,lift,lift_ep1400,coffee-pull,square}.npz` |
| AWR ckpt | `my_models/awr_s10000_<suite>.ckpt` (+ `awr_s10000_lift_ep1400.ckpt`) |
| MW BoN N-sweep | **full paths in § Table P N-sweep manifest** · `output/eval/matched_s10000/<mw>/bon_n{16,32}_n5/eval_log.json` · `logs/mw_*_bon16_32_*.log` · `logs/mw_*_bon32_only_*.log` · launchers `scripts/_run_mw_tablep_bon16_32.sh` · `_run_mw_tablep_bon32_only.sh` · `_queue_mw_bon16_32_wave2.sh` |
| RC TopK lock | `my_models/robocasa_<task>_topk_ep*_sr*.ckpt` + `…_topk_lock.txt` |
| Wave2 log | `logs/awr_s10000_<suite>_wave2_gpu{0\|1}.log` |
| Wave2 eval log | `logs/awr_s10000_<suite>_wave2_eval_gpu{0\|1}.log` |
| AWR eval | `output/eval/matched_s10000/<suite>/awr_n5/` |
| tmux (live) | `rc_wave1_close` / `rc_wave1_coffee` |
| ❌ never | `my_models/policy_awr_*` / old exploratory `awr_*.npz` / Table B |

Paper Wave1 artifacts:  
`output/eval/matched_s10000/<suite>/{baseline_n5,bon_n8_n5}/eval_log.json` + `summary.json` + `logs/matched_s10000_<suite>_gpu*.log`.

MW BoN N-sweep (locked):  
`output/eval/matched_s10000/{stick-pull,coffee-pull,disassemble,box-close}/bon_n{16,32}_n5/eval_log.json`  
+ `logs/mw_{stick,coffee}_bon16_32_*.log` · `logs/mw_{disassemble,box-close}_bon16_32_*.log` · `logs/mw_{disassemble,box-close}_bon32_only_*.log`.

**Lift** retrain **STOPPED**; run A matched **DONE** (incl. AWR + replan 8.0); run B Wave1+2 **DONE**.

**Имена файлов на suite (одинаковый шаблон):**

```text
output/eval/matched_s10000/<suite>/
  baseline_n5/eval_log.json     # Wave1 single-sample OAT8  ← paper source of truth
  bon_n8_n5/eval_log.json       # Wave1 BoN N=8 vote        ← primary paper BoN
  bon_n16_n5/eval_log.json      # MW N-sweep only (locked)
  bon_n32_n5/eval_log.json      # MW N-sweep only (locked)
  awr_n5/eval_log.json          # Wave2
  summary.json                  # protocol + SR + Δ + paths (N=8; N16/32 = eval_logs)
my_datasets/awr_s10000_<suite>.npz
my_models/awr_s10000_<suite>.ckpt
logs/matched_s10000_<suite>_gpu*.log           # Wave1 (DONE baseline/bon8)
logs/mw_*_bon16_32_*.log / mw_*_bon32_only_*.log  # MW N-sweep
logs/awr_s10000_<suite>_wave2_gpu*.log         # Wave2 collect+train
logs/awr_s10000_<suite>_wave2_eval_gpu*.log    # Wave2 AWR eval only
tmux: paper_s10000_<suite> | paper_w2_<suite> | rc_wave1_{close,coffee}
```

### Table P — Reproduce manifest (seeds + artifacts)

**Зачем:** один канон для reproduce / appendix — *на чём училось, чем выбирали ckpt, чем мерили paper SR*.  
Paths relative to `/workspace/oat`. Cluster: `oat_mipt_robomimic_askhabaliev_gs`.  
**Source of truth for numbers:** `matched_s10000/<suite>/summary.json` + `*/eval_log.json` (не train-time SR в имени ckpt).

#### Seed map (anti-leak — не смешивать)

| Role | Seed / range | Used for | ❌ Never use for |
|------|--------------|----------|------------------|
| **Train `cfg.seed` / `training.seed`** | RM **42**; MW specialists **0**; Lift *retrain* **7** | dataloader shuffle, weight init | paper SR |
| **Selection pool (TopK during policy train)** | env `test_start_seed=**1000**` (RM: 50 eps → `1000–1049`; MW: `n_test=250` → `1000–1249`) | pick `ep-*_sr-*.ckpt` | paper Table P / Δ |
| **Paper report pool** | `test_start_seed=**10000**`, `n_test=50` → episodes **`10000–10049`** | Wave1 baseline+BoN, Wave2 AWR eval, Table P | TopK / train selection |
| **AWR collect** | `--seed **0**` (worker offsets) | `my_datasets/awr_s10000_<suite>.npz` | selection `1000*` or report `10000*` |
| **Eval stochasticity** | `temperature=1.0`, `topk=10`; `-n 5` (=`num_exp`) | paper mean±std over 5 runs on **same** init set | — |
| **RoboCasa train seed** | **0** | tok + policy | — |
| **RoboCasa selection** | `test_start_seed=**2000**`, `n_test=50` → `2000–2049` | TopK / lock `robocasa_<task>_topk_lock.txt` | literal-5 report |
| **RoboCasa report (literal 5)** | seeds **`10000 10001 10002 10003 10004`**, each `-n 1 --n_test 50` | Wave1/2 SR mean±SEM | RM-style `-n 5` @ one start seed |
| **RoboCasa AWR collect** | `--seed **0**` | `awr` npz | selection 2000* / report 10000* |

**Inference (all paper evals):** OAT8 = `--use_k_tokens 8 --entropy_threshold 0`; BoN = `--bon_free 8 --bon_signal vote`.  
**Script:** `SUITE=<suite> BASE_CKPT=<path> [SKIP_AWR=1] GPU=<g> bash scripts/cluster_matched_triplet.sh`  
→ writes `output/eval/matched_s10000/<suite>/` + `logs/matched_s10000_<suite>_gpu*.log`.

#### Per-suite train → paper base (what trained on what)

| Suite | Data (zarr) | Tokenizer (frozen) | Policy run dir | Train seed | Paper **BASE_CKPT** (TopK @1000) | Matched root | AWR npz / ckpt |
|-------|-------------|--------------------|----------------|------------|----------------------------------|--------------|----------------|
| **Can** | `data/robomimic/can_N200.zarr` | `output/20260705/210939_train_oattok_can_N200/checkpoints/ep-0520_mse-0.005.ckpt` | `output/20260706/173343_train_oatpolicy_can_N200/` | 42 | `.../checkpoints/ep-1700_sr-0.940.ckpt` | `matched_s10000/can/` | `my_datasets/awr_s10000_can.npz` · `my_models/awr_s10000_can.ckpt` |
| **stick-pull** | `data/metaworld/stick-pull_N50.zarr` | `output/20260710/235437_train_oattok_mw-stick-pull_st_N50/checkpoints/ep-3030_mse-0.042.ckpt` | `output/20260711/134439_train_oatpolicy_mw-stick-pull_st_N50/` | 0 | `.../checkpoints/ep-0800_sr-0.212.ckpt` | `matched_s10000/stick-pull/` | `awr_s10000_stick-pull.npz` · `.ckpt` |
| **disassemble** | `data/metaworld/disassemble_N50.zarr` | `output/20260710/235437_train_oattok_mw-disassemble_st_N50/checkpoints/ep-3410_mse-0.027.ckpt` | `output/20260711/134440_train_oatpolicy_mw-disassemble_st_N50/` | 0 | `.../checkpoints/ep-1400_sr-0.700.ckpt` | `matched_s10000/disassemble/` | `awr_s10000_disassemble.npz` · `.ckpt` |
| **box-close** | `data/metaworld/box-close_N50.zarr` | `output/20260710/212943_train_oattok_mw-box-close_st_N50/checkpoints/ep-3450_mse-0.019.ckpt` | `output/20260711/134439_train_oatpolicy_mw-box-close_st_N50/` | 0 | `.../checkpoints/ep-2000_sr-0.552.ckpt` | `matched_s10000/box-close/` | `awr_s10000_box-close.npz` · `.ckpt` |
| **Lift** *(retrain DONE; **2 matched runs**)* | `data/robomimic/lift_N200.zarr` + HDF5 `data/robomimic/hdf5_datasets/lift_mh_image.hdf5` | `output/20260704/203215_train_oattok_lift_N200/checkpoints/ep-1970_mse-0.006.ckpt` | **retrain** `output/20260719/144024_train_oatpolicy_lift_N200/` | **7** | **TopK lock** `.../ep-1400_sr-0.950.ckpt`; mid-train Wave1 used `ep-0900_sr-0.930.ckpt` | **A:** `matched_s10000/lift/` (ep0900) · **B:** `matched_s10000/lift_ep1400/` (ep1400, paper) | **A:** `awr_s10000_lift.*` · **B:** `awr_s10000_lift_ep1400.*` |
| **Square** *(Wave1+2 DONE)* | `data/robomimic/square_N200.zarr` + `.../square_mh_image.hdf5` | `output/20260706/005048_train_oattok_square_N200/checkpoints/ep-0690_mse-0.004.ckpt` | `output/20260720/215024_train_oatpolicy_square_N200/` (seed=42; train killed on plateau) | 42 | **`.../ep-0700_sr-0.420.ckpt`** (lock `my_models/square_topk_lock.txt`) | `matched_s10000/square/` · base **36.0±7.5%** · BoN **28.0±9.4%** (Δ=**−8.0**) · AWR **31.2±5.2%** (Δ=**−4.8**) · replan probe **20.4** | `my_datasets/awr_s10000_square.npz` · `my_models/awr_s10000_square.ckpt` |
| **coffee-pull** *(TopK locked; Wave1 DONE → Wave2)* | `data/metaworld/coffee-pull_N50.zarr` | `output/20260710/212943_train_oattok_mw-coffee-pull_st_N50/checkpoints/ep-2670_mse-0.039.ckpt` | refit `output/20260720/090816_train_oatpolicy_mw-coffee-pull_st_N50/` (seed=0) | 0 | **`.../ep-1000_sr-0.432.ckpt`** (lock `my_models/coffee_pull_topk_lock.txt`) | `matched_s10000/coffee-pull/` · base **40.8±2.3%** · BoN **43.2±4.8%** (Δ=+2.4) · replan **12.3** | Wave2 collect **RUNNING** · tmux `coffee_w2` |

**RoboCasa** suites are **not** on the RM/MW `-n 5` layout — see **§ RoboCasa Table P** below (literal 5 seeds).

#### Lift retrain — reproduce lock (STOPPED 2026-07-21, plateau)

| Field | Value |
|-------|--------|
| Status | **STOPPED** @ ~ep **2332** (tmux `lift_retrain_resume` killed); train TopK plateau after **ep-1400** |
| Run dir | `output/20260719/144024_train_oatpolicy_lift_N200/` |
| Hydra | `.hydra/{overrides,config,hydra}.yaml` |
| Train log | `logs/train_policy_lift_retrain_s7_n100.log` |
| Launch / resume scripts | `scripts/cluster_policy_lift_retrain.sh` · `scripts/cluster_policy_lift_retrain_resume.sh` |
| `cfg.seed` / `training.seed` | **7** |
| `training.num_demo` | **200** |
| `training.rollout_every` | **100** |
| Train-time eval | `n_test=**100**`, `n_parallel_envs=4`, selection pool `test_start_seed=**1000**` (default) → **not** paper seed |
| Checkpoint TopK | `k=3`, `monitor_key=mean_success_rate` |
| Frozen tokenizer | `output/20260704/203215_train_oattok_lift_N200/checkpoints/ep-1970_mse-0.006.ckpt` |
| Data | `data/robomimic/lift_N200.zarr` + `data/robomimic/hdf5_datasets/lift_mh_image.hdf5` |
| **Paper BASE_CKPT (train TopK lock)** | `output/20260719/144024_train_oatpolicy_lift_N200/checkpoints/ep-1400_sr-0.950.ckpt` (**95.0%** @ selection pool) |
| Other TopK kept | `ep-0600_sr-0.910` · `ep-0900_sr-0.930` · `ep-1000_sr-0.900` · `ep-1200_sr-0.930` · `ep-1700_sr-0.940` · `latest.ckpt` (~stopped epoch) |
| TopK trajectory | 600→0.91, 900→0.93, 1000→0.90, 1200→0.93, **1400→0.95**, 1700→0.94; then +~900 ep no better |
| **Matched runs (2)** | Same protocol `test_start_seed=10000`, `n_test=50`, `-n 5`, OAT8, BoN N=8 vote. **Not** cherry-pick by Δ — run B = protocol TopK lock; run A = early Wave1 mid-train. |
| — **Run A** (ep-0900, mid-train) | Base `ep-0900_sr-0.930.ckpt` → `output/eval/matched_s10000/lift/` · baseline **82.0±3.2%** · BoN8 **84.8±4.6%** (Δ=+2.8) · AWR **81.2±4.1%** (Δ=−0.8) · `awr_s10000_lift.*` · **DONE** |
| — **Run B** (ep-1400, TopK lock / paper) | Base `ep-1400_sr-0.950.ckpt` → `output/eval/matched_s10000/lift_ep1400/` · **baseline 87.6±2.2%** · **BoN8 84.0±4.9%** (Δ=**−3.6**) · Wave2: npz+`awr_s10000_lift_ep1400.ckpt` **trained**; AWR eval **RUNNING** · replan **9.3** DONE · tmux `lift_awr_eval` |
| Anti-leak | **do not** cite train filename `sr-0.950` as paper SR; paper = `matched_s10000` only. Table P primary = **run B (ep-1400)** when Wave2 AWR finishes; run A kept for transparency. |

**tmux (live, ~2026-07-22 16:25 MSK) — parallel:** `lift_awr_eval` (GPU0, Lift B AWR eval) · `coffee_w2` (GPU0, coffee-pull Wave2 collect ~20%) · `square_matched` (GPU1, Square Wave1 baseline) · `rc_pol_chain` (GPU1, RC coffee_press resume @ ~ep300 / train-eval 39/50). Launchers: `scripts/_launch_lift_ep1400_awr_eval.sh` · `_launch_coffee_w2_parallel.sh` · `_launch_square_matched_parallel.sh` · `_launch_rc_coffee_resume.sh`.

**Square plateau watcher:** `scripts/_launch_square_matched_on_plateau.sh` — **FIRED** 2026-07-22T10:11Z → lock `ep-0700_sr-0.420.ckpt` (`my_models/square_topk_lock.txt`); train killed; Wave1 now in `square_matched`.

**DONE suite matched layout (example Can):**

```text
output/eval/matched_s10000/can/
  baseline_n5/eval_log.json
  bon_n8_n5/eval_log.json
  awr_n5/eval_log.json
  summary.json          # base_ckpt, awr_ckpt, protocol seeds, SR, Δ
  latency.json          # Table C
  latency_fair_kv.json  # Table C′
logs/matched_s10000_can_gpu0.log
```

**❌ Do not cite as paper:** `output/eval/matched/` (seed 1000 Table B); exploratory `my_models/policy_awr_*`; old deleted square/coffee `matched_s10000` from before 2026-07-20 reset; train-time `sr-0.XXX` in ckpt filename.

**Роль RM/MW в статье (к сводке от 2026-06-23):** LIBERO несёт диагноз (компаундинг / K·R null / отбор+).  
Наш трек = **ширина**: тот же позитив отбора (BoN + потом AWR) на RoboMimic/MetaWorld при **чистом** matched-протоколе. Не Таблица VI absolute parity.

### Latency / Table C — протокол репрезентативности (зафиксировано 2026-07-17)

Latency **нужна для статьи** (BoN = N× AR/replan; AWR = single-forward).  
Отдельный **policy-forward** замер — **не** перепрогон matched SR. Цифры Table P не меняются.

| | |
|--|--|
| **Когда** | после Wave 1–2 SR для suite’ов в paper (Phase 5) |
| **Что** | Single (OAT8) / BoN N=8 vote / AWR — **ms per policy call**, batch=1 |
| **Куда** | `output/eval/matched_s10000/<suite>/latency.json` → Table C (SR из Table P × ms × ΔSR × cost) |
| **Скрипт** | `my_scripts/measure_latency*` (+ `bon_free` + AWR ckpt), тот же docker/кластер, что paper eval |
| **Канон** | также [`RESOLUTIONPLAN.md`](RESOLUTIONPLAN.md) § Latency / Table C |

**Обязательно (иначе не в статью):**

1. **Тот же hardware / окружение**, где интерпретируем SR — один GPU того же типа (cluster V100), тот же docker/CUDA/driver. ❌ Mac/другая машина + cluster SR. ❌ TRT/ONNX только если так же крутится sim-eval.
2. **Batch=1 + deterministic** насколько возможно (`cudnn.deterministic` / fixed timing seeds). Warmup **вне** статистики.
3. **5–10 timed reps** на режим → в Table C **median** + **mean±std** (или IQR). ❌ один случайный прогон.
4. **Тот же input pipeline**, что sim-eval (resize/norm/To-stack/dtype) — obs из checkpoint dataset / val, не dummy, если он меняет путь. Single = OAT8 (`k=8`, `entropy_threshold=0`, T=1, topk=10); BoN = `bon_free=8` vote; AWR = `awr_s10000_<suite>.ckpt` single-sample.
5. **AWR ckpt = Wave2 артефакт, архитектура не менялась** — тот же `my_models/awr_s10000_<suite>.ckpt`, что дал `awr_n5/` (path/mtime/sha ↔ Wave2 log / summary). Baseline/BoN — тот же `BASE_CKPT`, что Wave1. ❌ exploratory `policy_awr_*` / другой epoch / ckpt после рефактора policy. Если код модели после Wave2 eval менялся — откат к commit Wave2 или пересчёт SR, не смешивать.
6. **`latency.json` обязан содержать воспроизводимость:** `git_commit`, `git_dirty`, `git_branch`, `measured_at`, `host`/`gpu_name`, `docker_image`, пути (+ optional sha256) `base_ckpt`/`awr_ckpt`, N reps + median/mean/std. Без commit — не paper-final.

**Не делать:** пересчёт Table P SR ради latency; мешать MuJoCo/render wall-clock в «policy latency» (с оговоркой, что episode time доминирует sim); кросс-suite SOTA latency без одной машины/протокола; AWR latency на другом ckpt/архитектуре, чем Wave2 Table P.

В тексте: *latency on same cluster GPU/stack as matched eval; batch=1; median over N timed forwards; same obs pipeline; same Wave2 AWR ckpt; code commit in latency.json; SR from Table P unchanged.*

### Table C — PAPER latency (paper-proof, current valid subset)

Источник: `output/eval/matched_s10000/<suite>/latency.json` + сводка `matched_s10000/table_c.json` (`paper_proof: true`).  
**Специфика замера (2-й / paper-proof прогон):** batch=1; warmup=20 excluded; **10 timed reps** → median; obs = val dataset; **obs counter reset + warmup per mode** (fair Single↔BoN↔AWR); `cudnn.deterministic`; policy-forward only (**не** MuJoCo).  
HW: **Tesla V100-SXM2-32GB**, torch **2.5.1+cu124**.  
**Git в артефакте:** docker без `.git` → `OAT_GIT_*` env при запуске (метка кода, не commit/push).  
- current valid subset: `git_commit=643bca01…` (2026-07-17 remasure)  
`git_dirty=true`. `square` and `coffee-pull` latency artifacts were reset pending rerun. Pre-proof (1-й прогон, не в статью): `matched_s10000/_latency_pre_paperproof/`.

| Suite | Single median | BoN N=8 median | AWR median | SR (Table P) base→BoN→AWR |
|-------|---------------|----------------|------------|---------------------------|
| Can | **40.9** ms | **40.3** ms | **40.4** ms | 76.4 → 80.8 → 79.2 |
| stick-pull | **47.2** | **46.2** | **49.9** | 15.6 → 25.6 → 26.8 |
| disassemble | **47.6** | **49.1** | **49.2** | 62.4 → 63.2 → 69.6 |
| box-close | **48.8** | **48.1** | **48.0** | 59.6 → 66.4 → 72.8 |

**Table C artifacts** (paper latency sources; все под `output/eval/matched_s10000/`):

| Suite | `latency.json` | AWR ckpt (timed) | Base ckpt (Single/BoN) | note |
|-------|----------------|------------------|------------------------|------|
| Can | `can/latency.json` | `my_models/awr_s10000_can.ckpt` | Wave1 `base_ckpt` in json | `paper_proof: true`, git `643bca01` |
| stick-pull | `stick-pull/latency.json` | `my_models/awr_s10000_stick-pull.ckpt` | idem | |
| disassemble | `disassemble/latency.json` | `my_models/awr_s10000_disassemble.ckpt` | idem | |
| box-close | `box-close/latency.json` | `my_models/awr_s10000_box-close.ckpt` | idem | |
| **сводка** | `table_c.json` | — | — | current valid subset only; rebuild: `scripts/build_table_c.py` |
| fair-KV rebuttal (**LOCKED**) | `<suite>/latency_fair_kv.json` | same AWR ckpts | same bases | `fair_kv: true`; `trials=8`; `batch=1`; сводка `table_c_fair_kv.json` |
| pre-proof archive | `_latency_pre_paperproof/<suite>_latency.json` | — | — | Table C only; **не в статью** |

Scripts: `scripts/measure_latency_paper.py`, `scripts/cluster_latency_paper_done.sh` (requires `OAT_GIT_COMMIT`), `scripts/build_table_c.py`.

**Read for paper:** on the current valid subset, BoN median ≈ Single (~40–49 ms) — vision encode **один раз** (amortized), AR дешёвый → N=8 почти не бьёт policy-forward cost. AWR ≈ Single. Table C = inference cost only.

**Table C′ — fair-KV rebuttal** — **не** заменяет Table C. Single / AWR\* = `predict_action` (KV-cache); BoN N = `predict_action_bon_free` / `generate` (KV).  
**Репрезентативно:** batch=1; 8 trials × 10 timed reps; paper = **mean±std of per-trial medians** (ms); obs=val; obs reset each trial; V100. Δ ≲ trial std → не «быстрее».  
**Колонки:** Single · BoN8 · BoN16 · BoN32 · **AWR8** (BoN8-distill) · **AWR16** (BoN16-distill @100ep).  
**Locked subset (2026-07-24):** Single / BoN8 / AWR8 на 7× RM+MW — `latency_fair_kv.json` + `table_c_fair_kv.json` (`paper_locked: true`, `git_commit=38455fbc…`). **Не перезаписывать** без явного remasure.  
**Fill (2026-07-28):** RC×4 Single+BoN{8,16,32} **DONE** → `matched_s10000/robocasa/<task>/latency_fair_kv_n16.json`. RM/MW BoN16/32 → same filename under `matched_s10000/<suite>/` (Single/BoN8/AWR8 stay in locked `latency_fair_kv.json`). **AWR16 DONE** (RoboMimic `can/lift/square` + RoboCasa `coffee_press_button`); **still open — RoboCasa only:** `close_drawer` / `turn_off_sink_faucet` / `turn_off_microwave` — [`AGENT_GUIDE_AWR16_LATENCY_REMAINING.md`](AGENT_GUIDE_AWR16_LATENCY_REMAINING.md).

| Suite | artifact (C′) | AWR8 ckpt | AWR16 ckpt | Base (Single/BoN\*) |
|-------|---------------|-----------|------------|---------------------|
| Can | `…/can/latency_fair_kv.json` + `latency_fair_kv_n16.json` | `awr_s10000_can.ckpt` | `Mirageinv/AWR` `robomimic_can_awr_bon16_e100.ckpt` | Wave1 `base_ckpt` in json |
| Lift | `…/lift/…` | `awr_s10000_lift.ckpt` | `…_lift_awr_bon16_e100.ckpt` | idem |
| Square | `…/square/…` | `awr_s10000_square.ckpt` | `…_square_awr_bon16_e100.ckpt` | idem |
| coffee-pull | `…/coffee-pull/…` | `awr_s10000_coffee-pull.ckpt` | TBD | idem |
| stick-pull | `…/stick-pull/…` | `awr_s10000_stick-pull.ckpt` | TBD | idem |
| disassemble | `…/disassemble/…` | `awr_s10000_disassemble.ckpt` | TBD | idem |
| box-close | `…/box-close/…` | `awr_s10000_box-close.ckpt` | TBD | idem |
| coffee_press_button | `…/robocasa/coffee_press_button/latency_fair_kv_n16.json` | — (RC = AWR16 track) | `Mirageinv/AWR` `robocasa_coffee_press_button_awr_bon16_e100.ckpt` | `robocasa_coffee_…_sr0.600.ckpt` |
| close_drawer | `…/robocasa/close_drawer/latency_fair_kv_n16.json` | — | TBD (not on HF yet) | `robocasa_close_…_sr0.700.ckpt` |
| turn_off_sink_faucet | `…/robocasa/turn_off_sink_faucet/latency_fair_kv_n16.json` | — | TBD (not on HF yet) | `robocasa_…_sink_…_sr0.580.ckpt` |
| turn_off_microwave | `…/robocasa/turn_off_microwave/latency_fair_kv_n16.json` | — | TBD (not on HF yet) | `robocasa_…_microwave_…_sr0.620.ckpt` |
| **сводка** | `matched_s10000/table_c_fair_kv.json` (locked) + per-suite `*_n16.json` | — | — | `fair_kv: true` |

| Suite | Single (KV) | BoN8 | BoN16 | BoN32 | AWR8 (KV) | AWR16 (KV) | BoN8−Single |
|-------|-------------|------|-------|-------|-----------|------------|-------------|
| **RoboMimic** | | | | | | | |
| Can | **40.6±1.4** | **43.3±2.1** | **42.9±1.7** | **41.7±1.6** | **42.5±2.4** | **42.9±2.0** | **+2.8** |
| Lift | **40.9±1.8** | **43.0±1.9** | **43.8±1.3** | **45.7±1.9** | **40.6±1.6** | **42.3±2.2** | **+2.1** |
| Square | **43.6±1.9** | **43.5±2.0** | **45.6±5.5** | **45.4±2.0** | **42.4±1.6** | **40.8±1.1** | **−0.1** |
| **MetaWorld** | | | | | | | |
| coffee-pull | **49.2±1.4** | **49.2±2.1** | **48.2±1.7** | **50.2±2.4** | **50.0±2.5** | TBD | **−0.0** |
| stick-pull | **46.6±2.0** | **47.4±2.1** | **51.7±2.4** | **52.4±2.9** | **45.7±1.3** | TBD | **+0.8** |
| disassemble | **46.3±1.9** | **48.5±1.0** | **49.5±1.7** | **50.2±2.5** | **45.6±0.5** | TBD | **+2.2** |
| box-close | **48.2±3.5** | **50.9±1.9** | **50.0±2.1** | **49.9±1.8** | **49.6±1.3** | TBD | **+2.7** |
| **RoboCasa** | | | | | | | |
| coffee_press_button | **45.5±2.0** | **45.5±2.4** | **47.0±4.0** | **46.6±2.1** | — | **43.6±1.7** | **+0.0** |
| close_drawer | **46.9±1.3** | **46.5±1.7** | **46.9±2.0** | **48.0±2.2** | — | TBD | **−0.4** |
| turn_off_sink_faucet | **44.1±1.9** | **44.1±1.7** | **45.5±2.4** | **47.2±2.2** | — | TBD | **+0.0** |
| turn_off_microwave | **45.0±1.9** | **48.1±1.3** | **47.1±2.6** | **46.7±2.5** | — | TBD | **+3.1** |

В тексте: main = Table C (deployed); appendix = Table C′ (fair KV). Read: Single≈BoN8≈AWR8≈AWR16 ~41–51 ms; BoN16/32 overhead usually within noise / ≲5 ms (stick-pull rises more). **AWR16 DONE** for HF set (can/lift/square/coffee); **RC only** close/sink/mw AWR16 — нет ckpt на HF. Locked Single/BoN8/AWR8: do **not** overwrite `latency_fair_kv.json`.

### Replan count probe (lab, 2026-07-21) — не paper-final

**Зачем:** episode cost ≈ `(mean_replans) × (ms/call from Table C)`. Table C меряет только forward; runners теперь логируют `mean_replans_per_episode` / `std` / `max` при каждом `eval_policy_sim`. При фиксированном `R=16` (OAT default) baseline/BoN/AWR дают **одинаковое** число replan на suite — достаточно одного baseline-style прогона.

**Протокол probe (не matched):** OAT8 (`--use_k_tokens 8 --entropy_threshold 0`), `-n 1`, `--n_test 10`, `--n_parallel_envs 1`, `MUJOCO_GL=egl`, cluster V100. Default env seed base (=1000). **Не** заменяет matched `n_test=50` / Table P.

| Suite | Status | mean_replans | std | max | SR (probe) | ckpt | tmux (was) · Artifacts |
|-------|--------|--------------|-----|-----|------------|------|------------------------|
| **MW box-close** | **DONE** | **10.6** | 2.62 | 13 | 0.50 | `output/20260711/134439_train_oatpolicy_mw-box-close_st_N50/checkpoints/ep-2000_sr-0.552.ckpt` | **`replan_can_box`** · `eval_out/replan_probe_box-close/` (`eval_log.json`) · `logs/replan_probe_box-close.log` |
| **MW disassemble** | **DONE** | **9.7** | 3.55 | 13 | 0.60 | `output/20260711/134440_train_oatpolicy_mw-disassemble_st_N50/checkpoints/ep-1400_sr-0.700.ckpt` | **`replan_disassemble`** · `eval_out/replan_probe_disassemble/` (`eval_log.json`) · `logs/replan_probe_disassemble.log` |
| **MW stick-pull** | **DONE** | **12.6** | 1.2 | 13 | 0.10 | `output/20260711/134439_train_oatpolicy_mw-stick-pull_st_N50/checkpoints/ep-0800_sr-0.212.ckpt` | **`replan_stick_pull`** · `eval_out/replan_probe_stick-pull/` (`eval_log.json`) · `logs/replan_probe_stick-pull.log` |
| **RoboMimic Can** | **DONE** | **10.4** | 1.96 | 14 | 1.00 | `.../ep-1700_sr-0.940.ckpt` | extract DONE `can_mh_image.hdf5`; **`replan_can`** · `eval_out/replan_probe_can/` |
| **Lift ep-0900** | **DONE** | **8.0** | 3.35 | 14 | 1.00 | `.../ep-0900_sr-0.930.ckpt` | `eval_out/replan_probe_lift900/` · `logs/replan_probe_lift900.log` |
| **MW coffee-pull** | **DONE** | **12.3** | 2.10 | 13 | 0.10 | `.../ep-1000_sr-0.432.ckpt` | `eval_out/replan_probe_coffee-pull/` · `logs/replan_probe_coffee-pull.log` |
| **Lift ep-1400** | **DONE** | **9.3** | 5.93 | 25 | 0.90 | `.../ep-1400_sr-0.950.ckpt` | `eval_out/replan_probe_lift1400/` · `logs/replan_probe_lift1400.log` |
| **RoboMimic Square** | **DONE** | **20.4** | 6.28 | 25 | 0.40 | `.../ep-0700_sr-0.420.ckpt` (TopK lock) | **`replan_square`** · `eval_out/replan_probe_square/` · `logs/replan_probe_square.log` · Wave1 corroborate **21.8** @ `matched_s10000/square/baseline_n5/` |

**Note:** replan probe uses **BASE** ckpt only — independent of BoN/AWR; safe to run before/during Wave1–2. Chain `scripts/_launch_replan_lift_coffee_chain.sh` **ALL DONE** 2026-07-22T00:50Z. Square twin probe **DONE** 2026-07-23T16:29Z (`scripts/_launch_replan_square.sh`).

**Square — highest replans + worst BoN (interpretation, 2026-07-23):**

| signal | Square | rest of Table P |
|--------|--------|-----------------|
| mean_replans (n=10 probe) | **20.4** (Wave1 matched **21.8**) | **8.0–12.6** |
| max replans | **25** (= horizon 400 / R=16) | MW ≤13; Lift1400 also max 25 but mean **9.3** |
| Δ_BoN | **−8.0** (worst) | mostly **+0.8…+10**; only Lift B also negative (−3.6) |
| Δ_AWR | **−4.8** (partial recover vs BoN) | mostly positive |

1. **Why so many replans:** Square horizon **400** → ceiling 25 calls/ep at R=16. Matched SR only **0.36** → most episodes do **not** finish early → mean sits near the ceiling (probe 20.4 / Wave1 21.8; BoN Wave1 replan **22.5** same regime). Contrast Lift1400: same horizon/max, but SR≈0.9 → mean only **9.3**. So high replan here is **failure length**, not a different controller.
2. **Why BoN is worst here:** verifier-free `vote` = mode-seeking / outlier rejection. On a **weak** policy with **~22 decisions/ep**, a bad consensus pick compounds over the whole episode. Suites where BoN helps (Can, box, stick) have either higher competence or fewer replans → less room for compounded mode error. Lift B (Δ_BoN −3.6 @ replan 9.3) is the mild version of the same pattern; Square is the extreme.
3. **AWR:** distills BoN into single-sample → **31.2** (between BoN 28 and base 36) — recovers ~half the BoN damage, **cannot** beat base. Consistent with “vote was anti-informative on this suite”: baking the vote mode still sits below the raw OAT8 policy.
4. **Paper takeaway:** BoN is **not** uniformly +SR; the largest replan budget (longest effective open-loop chain) co-occurs with the largest **negative** Δ_BoN. Episode compute cost is also highest on Square (`replans × ms/call`). Report Square as the **failure-mode** of test-time vote under long-horizon / low-SR, not an anomaly to hide.

**Read (n=10 probe):** MW ≈ **9.7–12.6**; Can **10.4**; Lift900 **8.0** / Lift1400 **9.3**; **Square 20.4 = outlier high** (≈2× next).

**Ключи в `eval_log.json`:** `mean_replans_per_episode_mean`, `std_replans_per_episode_mean`, `max_replans_per_episode_mean` (+ обычные SR/tokens).  
**Код:** `oat/env_runner/{metaworld,robomimic}_runner.py` (`episode_replans`); print в `scripts/eval_policy_sim.py`.  
**Launcher:** `scripts/_launch_replan_lift_coffee_chain.sh` · `scripts/_launch_replan_square.sh` · `scripts/_launch_replan_can.sh`.

**Мусор удалён:** `eval_out/replan_probe_{lift,mt4}` · early smoke logs. Kept: `replan_probe_{box-close,disassemble,stick-pull,can,lift900,coffee-pull,lift1400,square}`.

#### Finished matched snapshot (verified from `summary.json` / replan `eval_log.json`, 2026-07-22)

| Suite | train seed | TopK @1000 | report @10000 | baseline | BoN8 | AWR | Δ_BoN | Δ_AWR | roots |
|-------|------------|------------|---------------|----------|------|-----|-------|-------|-------|
| Can | 42 | ep-1700 | yes | 76.4±3.0 | 80.8±1.1 | 79.2±7.7 | +4.4 | +2.8 | `matched_s10000/can/` · `awr_s10000_can.*` |
| stick-pull | 0 | ep-0800 | yes | 15.6±6.2 | 25.6±2.6 | 26.8±5.2 | +10.0 | +11.2 | `matched_s10000/stick-pull/` · `awr_s10000_stick-pull.*` |
| disassemble | 0 | ep-1400 | yes | 62.4±5.2 | 63.2±6.3 | 69.6±5.7 | +0.8 | +7.2 | `matched_s10000/disassemble/` · `awr_s10000_disassemble.*` |
| box-close | 0 | ep-2000 | yes | 59.6±7.5 | 66.4±3.0 | 72.8±4.1 | +6.8 | +13.2 | `matched_s10000/box-close/` · `awr_s10000_box-close.*` |
| Lift A (ep-0900) | 7 | mid-train | yes | 82.0±3.2 | 84.8±4.6 | 81.2±4.1 | +2.8 | −0.8 | `matched_s10000/lift/` · `awr_s10000_lift.*` · replan **8.0** |
| Lift B (ep-1400) | 7 | ep-1400 | yes | **87.6±2.2** | **84.0±4.9** | AWR eval RUNNING | **−3.6** | — | `matched_s10000/lift_ep1400/` · `awr_s10000_lift_ep1400.*` · replan **9.3** |
| coffee-pull | 0 | ep-1000 | yes | **40.8±2.3** | **43.2±4.8** | Wave2 collect ~20% | **+2.4** | — | `matched_s10000/coffee-pull/` · replan **12.3** |
| Square | 42 | ep-0700 | yes | **36.0±7.5** | **28.0±9.4** | **31.2±5.2** | **−8.0** | **−4.8** | `matched_s10000/square/` · `awr_s10000_square.*` · replan probe **20.4** |
| Can replan probe | — | ep-1700 | n/a (seed 1000 probe) | — | — | — | — | — | mean_replans **10.4**±1.96 max14 SR1.0 · `eval_out/replan_probe_can/` |
| Lift900 replan | — | ep-0900 | n/a | — | — | — | — | — | mean_replans **8.0**±3.35 max14 SR1.0 · `eval_out/replan_probe_lift900/` |
| coffee-pull replan | — | ep-1000 | n/a | — | — | — | — | — | mean_replans **12.3**±2.10 max13 SR0.1 · `eval_out/replan_probe_coffee-pull/` |
| Lift1400 replan | — | ep-1400 | n/a | — | — | — | — | — | mean_replans **9.3**±5.93 max25 SR0.9 · `eval_out/replan_probe_lift1400/` |
| Square replan | — | ep-0700 | n/a | — | — | — | — | — | probe **20.4**±6.28 max25 SR0.40 · `eval_out/replan_probe_square/`; Wave1 **21.8** · `matched_s10000/square/baseline_n5/` |

### Table B — LAB ONLY (seed 1000, selection pool) — не paper

Deprecated for paper claims. Kept so we don't lose the draft numbers.

| Suite | baseline | BoN | AWR | note |
|-------|----------|-----|-----|------|
| Can | 79.3±1.2 | 80.0±5.3 | 76.0±0.0 | selection-pool leak |
| coffee | 30.0±4.0 | 30.0±4.0 | 29.3±1.2 | lab |
| stick | 18.7±10.1 | 25.3±3.1 | 30.7±6.4 | lab |
| disassemble | 69.3±7.0 | 70.0±6.0 | — | lab |
| box | 52.0±3.5 | 56.0±5.3 | — | was under `metaworld_box-close/` |
| Lift | 78.7±1.2 | — | — | retrain before paper |
| Square | **25.3±4.2** (ep-0600) | — | — | baseline seed1000 DONE; BoN/AWR not run on that pool |

Paths: `output/eval/matched/<suite>/` (seed 1000). **Do not cite in paper.**

---

## RoboCasa — Table P (literal 5 seeds) + reproduce

**Canonical protocol:** [`ROBOCASA.md`](ROBOCASA.md) · live log: [`RESULTS_ROBOCASA.md`](RESULTS_ROBOCASA.md).  
**Layout differs from RM/MW:** report = **5 distinct seeds** `10000…10004`, each `-n 1 --n_test 50` (not `-n 5` on one block). Aggregate = **mean ± SEM** (`SEM=SD/√5`).

### Seed / HP lock

| Role | Value |
|------|--------|
| Data | official RoboCasa v0.2 `human_im`+`mg_im` → **50H+150M**, subsample seed **0**, Da=**12** |
| Zarr | `data/robocasa/<task>_N200.zarr` (+ `ROBOCASA_SOURCE.txt` inside) |
| Train `training.seed` / `seed` | **0** (tok + policy) |
| Selection (TopK) | `test_start_seed=**2000**`, `n_test=50` → eps `2000–2049` |
| Report | seeds **`10000 10001 10002 10003 10004`**, `-n 1` each |
| AWR collect | `--seed 0` |
| **AWR train** | **`--epochs 100`** (RoboCasa lock) · `--beta 0.5 --beta_kl 0.05` |
| OAT8 / BoN | `--use_k_tokens 8 --entropy_threshold 0` · `--bon_free 8 --bon_signal vote` · `temp=1.0` `topk=10` |
| Policy TopK | `k=3`, `monitor_key=mean_success_rate`; lock file `my_models/robocasa_<task>_topk_lock.txt` |
| First sim-eval | `training.rollout_start_epoch=200` (skip useless ep-0) for **new** launches |
| Venv | `.venv_robocasa` (robosuite 1.5) — never shared `.venv` |
| Wave1 script | `SUITE=<task> BASE_CKPT=<ckpt> GPU=<g> bash scripts/cluster_robocasa_literal5_wave1.sh` |
| Wave2 AWR | `SUITE=<task> BASE_CKPT=<ckpt> GPU=<g> bash scripts/cluster_robocasa_literal5_wave2_awr.sh` (**epochs=100**) |
| Aggregate | `python scripts/aggregate_robocasa_literal5.py --root output/eval/matched_s10000/robocasa/<task>`

### Table P — RoboCasa (fill from `summary_literal5.json` only)

| task | BASE_CKPT (TopK @2000) | baseline mean±SEM | BoN N=8 | AWR | Δ_BoN±SEM_Δ | Δ_AWR±SEM_Δ | artifacts |
|------|------------------------|-------------------|---------|-----|-------------|-------------|-----------|
| close_drawer | **`my_models/robocasa_close_drawer_topk_ep0500_sr0.700.ckpt`** | **56.0±1.1%** | **59.6±1.6%** | — | **+3.6±1.9** | — | Wave1 **DONE** · `matched_s10000/robocasa/close_drawer/summary_literal5.json` · lock `…_topk_lock.txt` |
| coffee_press_button | **`my_models/robocasa_coffee_press_button_topk_ep0500_sr0.600.ckpt`** | **44.0±3.0%** | **55.6±2.7%** | — | **+11.6±4.1** | — | Wave1 **DONE** · `…/coffee_press_button/summary_literal5.json` · lock `…_topk_lock.txt` |
| turn_off_sink_faucet | **`my_models/robocasa_turn_off_sink_faucet_topk_ep0500_sr0.580.ckpt`** | **52.4±2.5%** | **56.0±3.4%** | — | **+3.6±4.2** | — | Wave1 **DONE** · `…/turn_off_sink_faucet/summary_literal5.json` · lock `…_topk_lock.txt` |
| turn_off_microwave | **`my_models/robocasa_turn_off_microwave_topk_ep0500_sr0.620.ckpt`** | **43.6±3.1%** (5/5: 0.46/0.52/0.46/0.40/0.34) | BoN **RUNNING** (1/5) | — | — | — | `matched_s10000/robocasa/turn_off_microwave/` · Wave1 live |

**Eval tree (per task):**
```text
output/eval/matched_s10000/robocasa/<task>/
  baseline_seed{10000..10004}/eval_log.json
  bon_n8_seed{10000..10004}/eval_log.json
  awr_seed{10000..10004}/eval_log.json          # Wave2
  summary_literal5.json
  wave1_literal5.log
```

### Reproduce manifest — data / tok / policy (cluster paths)

| task | zarr | `ROBOCASA_SOURCE.txt` sha256 | Tokenizer (frozen MSE top) | Policy run | Train status |
|------|------|------------------------------|----------------------------|------------|--------------|
| close_drawer | `data/robocasa/close_drawer_N200.zarr` (1.4G) | `b4a8219c740e0f58…723c2b` | `output/20260720/005709_train_oattok_close_drawer_N200/checkpoints/ep-1800_mse-0.002.ckpt` | **scratch** `output/20260724/220823_train_oatpolicy_close_drawer_N200/` | **STOPPED** · TopK lock ep-0500@0.700 · Wave1 live |
| coffee_press_button | `data/robocasa/coffee_press_button_N200.zarr` (809M) | `3cd0cfca695c55bb…734adac` | `output/20260720/041753_train_oattok_coffee_press_button_N200/checkpoints/ep-1940_mse-0.003.ckpt` | **scratch** `output/20260724/220823_train_oatpolicy_coffee_press_button_N200/` | **STOPPED** · TopK lock ep-0500@0.600 · Wave1 live |
| turn_off_microwave | `data/robocasa/turn_off_microwave_N200.zarr` (1.2G) | `183855a4c7e31588…28747a` | `output/20260720/061925_train_oattok_turn_off_microwave_N200/checkpoints/ep-2720_mse-0.002.ckpt` | *(none)* | **deferred** |
| turn_off_sink_faucet | `data/robocasa/turn_off_sink_faucet_N200.zarr` (1.2G) | `778f8e86853469d8…1eb7f8` | `output/20260720/083055_train_oattok_turn_off_sink_faucet_N200/checkpoints/ep-3080_mse-0.002.ckpt` | *(none)* | **deferred** |

**Lock → Wave1 (2026-07-26):** user stop-fit; TopK frozen from selection@2000; `FORCE_RERUN=1` literal-5 Wave1 on sibling `oat_mw_bon32_fix` (EGL: `MUJOCO_EGL_DEVICE_ID=0` after `CUDA_VISIBLE_DEVICES` remap).

**Deleted junk (2026-07-23, 0 TopK only):** empty coffee hydra `20260721/202439_*`; OOM microwave `20260723/041317_*microwave*`; killed sink `20260723/041317_*sink*`.

---

## RoboMimic — all DONE

Pipelines: chain5 (internal 250-episode estimate) + BoN N=8 + AWR for Lift, Can, Square. See summary table; Square BoN/AWR from **ep-0600** (not best chain5 ep-1500).

## RoboMimic — Lift

### Retrain (paper path) — STOPPED 2026-07-21

Primary policy fit for matched rematch. Full reproduce table: **«Lift retrain — reproduce lock»** above.

| Item | Value |
|------|-------|
| Run dir | `output/20260719/144024_train_oatpolicy_lift_N200/` |
| Seeds | `seed=7`, `training.seed=7` |
| Train eval | `n_test=100`, pool seed **1000**, `rollout_every=100` |
| **TopK-1 (lock)** | `checkpoints/ep-1400_sr-0.950.ckpt` |
| Stopped | ~ep **2332**; no TopK >0.95 after ep1400 |
| Train log | `logs/train_policy_lift_retrain_s7_n100.log` |
| Run A matched (ep-0900) | baseline **82.0±3.2%**, BoN8 **84.8±4.6%** → `matched_s10000/lift/` |
| Run B matched (ep-1400 / paper) | Wave1→Wave2 **QUEUED** → `matched_s10000/lift_ep1400/` · tmux `lift_ep1400_matched` |

### Data
| Artifact | Path |
|----------|------|
| HDF5 | `data/robomimic/hdf5_datasets/lift_mh_image.hdf5` |
| Zarr (200 demos) | `data/robomimic/lift_N200.zarr` |
| Gen / convert logs | `logs/download_lift.log`, convert via `scripts/prepare_robomimic_lift.sh` |

### Tokenizer
| Item | Value |
|------|-------|
| Run dir | `output/20260704/203215_train_oattok_lift_N200/` |
| Train log | `logs/train_tok_lift.log` |
| **Best ckpt (top-1 mse)** | `output/20260704/203215_train_oattok_lift_N200/checkpoints/ep-1970_mse-0.006.ckpt` |
| Config | `task/tokenizer=robomimic/lift`, `training.num_demo=200`, `num_epochs=5001`, top-3 by `test_reconst_mse` |

### Policy (exploratory / pre-retrain — not paper matched base)
| Item | Value |
|------|-------|
| Run dir (old paper s42) | `output/20260706/163500_train_oatpolicy_lift_N200/` |
| Train log | `logs/train_policy_lift_paper_s42.log` |
| **Chain5 + BoN/AWR ckpt** | `.../ep-0600_sr-0.920.ckpt` |
| Alt chain5 only (worse) | `.../ep-1200_sr-0.900.ckpt` |
| Frozen tokenizer | `ep-1970_mse-0.006.ckpt` (above) |

### Eval — Chain5 (250 eps)
| Ckpt | Output dir | Log | SR |
|------|------------|-----|-----|
| **ep-0600** (BoN/AWR base) | `output/eval/robomimic_lift_paper5_ep0600/` | `logs/eval_lift_chain5.log` | **83.6 ± 1.7%** |
| ep-1200 (extra run) | `output/eval/robomimic_lift_paper5_ep1200/` | `logs/eval_lift_chain5_ep1200.log` | 77.6 ± 3.1% |

Per-environment-block ep-0600: 0.84, 0.86, 0.78, 0.88, 0.82. Paper target: **99.2 ± 0.5%**.

### Eval — BoN → AWR pipeline
| Item | Value |
|------|-------|
| Pipeline script | `scripts/cluster_lift_bon_awr_pipeline.sh` |
| Pipeline log | `logs/lift_bon_awr_pipeline.log` |
| Status | **DONE** (2026-07-10 05:28 UTC) |
| Base ckpt | `ep-0600_sr-0.920.ckpt` |

**BoN**
| Item | Value |
|------|-------|
| Output | `output/eval/robomimic_lift_bon_n8_n3/` |
| Log artifact | `output/eval/robomimic_lift_bon_n8_n3/eval_log.json` |
| **SR** | **91.3 ± 2.9%** (exp: 0.92, 0.86, 0.96) |

**AWR collect / train**
| Item | Value |
|------|-------|
| Dataset | `my_datasets/awr_lift_bon.npz` |
| Collect source SR | per-episode **92.3%**, chunk-weighted baseline **77.8%** |
| AWR ckpt (saved path) | `my_models/policy_awr_lift.ckpt` |
| Train | 30 ep, `beta=0.5`, `beta_kl=0.05`, `ordering=uniform` |

**AWR eval**
| Item | Value |
|------|-------|
| Output | `output/eval/robomimic_lift_awr_n3/` |
| Log artifact | `output/eval/robomimic_lift_awr_n3/eval_log.json` |
| **SR** | **93.3 ± 1.8%** (exp: 0.94, 0.96, 0.90) |

---

## RoboMimic — Can

### Data
| Artifact | Path |
|----------|------|
| HDF5 (raw mh) | `data/robomimic/hdf5_datasets/can/mh/demo_v15.hdf5` |
| HDF5 (image) | `data/robomimic/hdf5_datasets/can/mh/image_v15.hdf5` |
| Zarr | `data/robomimic/can_N200.zarr` |
| Logs | `logs/download_can.log`, `logs/extract_can_mh_image.log`, `logs/convert_can.log` |

### Tokenizer
| Item | Value |
|------|-------|
| Run dir | `output/20260705/210939_train_oattok_can_N200/` |
| Train log | `logs/train_tok_can.log` |
| **Best ckpt** | `output/20260705/210939_train_oattok_can_N200/checkpoints/ep-0520_mse-0.005.ckpt` |

### Policy
| Item | Value |
|------|-------|
| Run dir | `output/20260706/173343_train_oatpolicy_can_N200/` |
| Train log | `logs/train_policy_can_paper_s42.log` |
| **Chain5 ckpt** | `output/20260706/173343_train_oatpolicy_can_N200/checkpoints/ep-1700_sr-0.940.ckpt` |
| Also in top-k | `ep-2100_sr-0.920.ckpt` |

### Eval — Chain5
| Item | Value |
|------|-------|
| Output dir | `output/eval/robomimic_can_paper5_ep1700/` |
| Log | `logs/eval_can_chain5.log` |
| Summary | `output/eval/robomimic_can_paper5_ep1700/summary.json` |
| **SR** | **86.0 ± 2.8%** (environment blocks: 0.90, 0.86, 0.86, 0.92, 0.76) |
| Paper target | 80.8 ± 2.3% |

### Eval — BoN → AWR pipeline
| Item | Value |
|------|-------|
| Pipeline script | `scripts/cluster_can_bon_awr_resume.sh` |
| Pipeline log | `logs/can_bon_awr_pipeline.log` |
| Status | **DONE** (2026-07-10 05:04 UTC) |
| Base ckpt | `ep-1700_sr-0.940.ckpt` |

**BoN:** `output/eval/robomimic_can_bon_n8_n3/` → **90.7 ± 1.3%** (`eval_log.json`)

**AWR:** dataset `my_datasets/awr_can_bon.npz` (collect per-ep SR **88.7%**); ckpt path `my_models/policy_awr_can.ckpt`; eval `output/eval/robomimic_can_awr_n3/` → **89.3 ± 1.8%**

Per-exp verification from `logs/can_bon_awr_pipeline.log`: BoN **92%, 92%, 88%**; AWR **90%, 92%, 86%**. The corresponding chain5 block on the same initializations (`1000–1049`) has one baseline sample at 90%; a matched baseline with 3 stochastic repeats is still required before treating +4.7/+3.3 pp as a final causal delta.

---

## RoboMimic — Square

### Data
| Artifact | Path |
|----------|------|
| HDF5 | `data/robomimic/hdf5_datasets/square/mh/` (extracted mh image) |
| Zarr | `data/robomimic/square_N200.zarr` |

### Tokenizer
| Item | Value |
|------|-------|
| Run dir | `output/20260706/005048_train_oattok_square_N200/` |
| Train log | `logs/train_tok_square.log` |
| **Best ckpt** | `output/20260706/005048_train_oattok_square_N200/checkpoints/ep-0690_mse-0.004.ckpt` |

### Policy
| Item | Value |
|------|-------|
| Run dir | `output/20260707/102446_train_oatpolicy_square_N200/` |
| Train log | `logs/train_policy_square_paper_s42.log` |
| Chain5 ckpt | rerun pending |
| **BoN base ckpt** | rerun pending |
| **Best 250-eps ckpt** | rerun pending |
| Latest top-k | `ep-2200_sr-0.440.ckpt` |

### Eval — Chain5
| Run | Output dir | Log | SR |
|-----|------------|-----|-----|
| ep-0600 | `output/eval/robomimic_square_paper5_ep0600/` | `logs/eval_square_chain5.log` | 30.8 ± 2.7% |
| **ep-1500** | `output/eval/robomimic_square_paper5_ep1500/` | `logs/eval_square_chain5_ep1500.log` | **31.2 ± 1.5%** |

Per-environment-block (ep1500): 0.32, 0.26, 0.30, 0.34, 0.34. Paper target: **39.2 ± 2.4%**.

### Eval — BoN → AWR pipeline
| Item | Value |
|------|-------|
| Pipeline script | rerun pending |
| Pipeline log | rerun pending |
| Status | reset on 2026-07-20; old exploratory artifacts deleted |
| Base ckpt | rerun pending |

### Eval — PAPER matched (`matched_s10000`, anti-leak) — Wave1+2 **DONE**

| Item | Value |
|------|-------|
| Policy run (paper) | `output/20260720/215024_train_oatpolicy_square_N200/` |
| **BASE_CKPT** | `.../checkpoints/ep-0700_sr-0.420.ckpt` · lock `my_models/square_topk_lock.txt` |
| Matched root | `output/eval/matched_s10000/square/` · `summary.json` |
| Wave1 baseline | **36.0±7.5%** · `baseline_n5/eval_log.json` |
| Wave1 BoN N=8 | **28.0±9.4%** · `bon_n8_n5/eval_log.json` · Δ=**−8.0** (worst suite) |
| Wave2 AWR | **31.2±5.2%** · `awr_n5/eval_log.json` · Δ=**−4.8** · `awr_s10000_square.{npz,ckpt}` |
| **Replan (canonical n=10)** | **20.4** ±6.28 / max **25** / probe SR 0.40 · `eval_out/replan_probe_square/` |
| Replan (Wave1 matched) | **21.8** ±4.84 / max 25 · `baseline_n5/eval_log.json` (corroborates probe) |

**Interpretation:** highest replan count among all suites **and** largest negative Δ_BoN — see replan § “Square — highest replans + worst BoN”. AWR only partially undoes BoN damage; neither beats base. Cite as BoN failure mode under long-horizon / low-SR compounding.

Old exploratory matched/AWR/latency (pre-2026-07-20) deleted; cite only paths above.

---

## MetaWorld — MT4 multitask (DONE)

One shared model on interleaved `mt4_N50.zarr` (200 eps, 50/task). **Not** paper single-task specialist protocol.

### Data
| Artifact | Path |
|----------|------|
| Zarr | `data/metaworld/mt4_N50.zarr` |
| Gen log | `logs/gen_metaworld_mt4_N50.log` |

### Tokenizer
| Item | Value |
|------|-------|
| Run dir | `output/20260707/124135_train_oattok_mw-mt4_N50/` |
| Train log | `logs/train_oattok_mt4_N50.log` |
| **Best ckpt (used by policy)** | `output/20260707/124135_train_oattok_mw-mt4_N50/checkpoints/ep-3830_mse-0.024.ckpt` (exact `test_reconst_mse` **0.0235** @ ep-3830; `ep-4010_mse-0.024.ckpt` also on disk, unused) |

### Policy
| Item | Value |
|------|-------|
| Run dir | `output/20260708/032431_train_oatpolicy_mw-mt4_N50/` |
| Train logs | `logs/train_policy_mt4_paper_s0.log`, `logs/train_policy_mt4_paper_s42.log` |
| Train eval | fast mode during train (`n_test=50`, `rollout_every=50`) |
| Frozen tokenizer | `ep-3830_mse-0.024.ckpt` (above; set in `cluster_policy_mt4_paper.sh`) |
| **Chain5 / BoN ckpt** | `.../checkpoints/ep-0450_sr-0.280.ckpt` |
| Other top-k | `ep-0200_sr-0.280.ckpt`, `ep-1000_sr-0.240.ckpt` |

### Eval — Chain5 (250 eps, 4 tasks interleaved)
| Item | Value |
|------|-------|
| Output dir | `output/eval/metaworld_mt4_paper5_ep0450/` |
| Log | `logs/eval_mt4_chain5.log` |
| Summary | `output/eval/metaworld_mt4_paper5_ep0450/summary.json` |
| **Overall SR** | **28.4 ± 3.1%** |
| **Mean of 4 task SRs** | **27.7%** |

Per-task SR (chain5):

| Task | Ours | Paper Table VI |
|------|------|----------------|
| box-close | 38.5% | 44.4% |
| coffee-pull | 50.8% | 26.4% |
| disassemble | 21.7% | 17.2% |
| stick-pull | **0.0%** | 9.6% |
| average | 27.7% | 24.4% |

### Eval — BoN → AWR pipeline
| Item | Value |
|------|-------|
| Pipeline script | `scripts/cluster_mt4_bon_awr_pipeline.sh` |
| Pipeline log | `logs/mt4_bon_awr_pipeline.log` |
| Status | **DONE** (2026-07-10 23:42 UTC) |
| Base ckpt | `ep-0450_sr-0.280.ckpt` |

**BoN** — `output/eval/metaworld_mt4_bon_n8_n3/eval_log.json`
- Overall: **26.7 ± 2.4%** (exp: 28%, 30%, 22%)
- Per-task: box-close 48.7%, coffee-pull 46.2%, disassemble 8.3%, stick-pull 0%

**AWR collect** — `my_datasets/awr_mt4_bon.npz`
- 20k chunks / 1693 episodes, 4 tasks. Distinguish two success rates (same labels, different weights):
  - **per-episode SR 16.6%** (mean over episodes; each episode contributes once)
  - **chunk-weighted SR 8.4%** (mean over chunks; longer failed episodes pull this down)
- Per-task **per-episode** SR: box 25.5%, coffee 30.7%, disassemble 6.0%, stick 0%
- Per-task **chunk-weighted** SR: box 15.4%, coffee 14.7%, disassemble 3.4%, stick 0% (weak BoN source vs LIBERO ~72%)

**AWR train** → `my_models/policy_awr_mt4.ckpt` (845 MB, 30 ep)

**AWR eval** — `output/eval/metaworld_mt4_awr_n3/eval_log.json`
- Overall: **18.7 ± 1.8%** (exp: 22%, 18%, 16%; regression vs baseline)
- Per-task: box-close 41.0%, coffee-pull 30.8%, disassemble 0%, stick-pull 0%

**Verdict:** no SR gain on MT4 multitask; BoN flat/−2 pp, AWR −10 pp (expected with weak BoN collect: 16.6% per-ep / 8.4% chunk-weighted).

---

## MetaWorld — single-task specialists

Our specialist setup: one model per task, 50 demos, **full** train-time sim eval (`n_test=250`, `rollout_every=200`, top-3 SR ckpts). Chain5 = one selected checkpoint evaluated on 5 environment-seed blocks × 50 = 250 episodes (eval variance for a **fixed** policy). The paper’s ± comes from 5 independently trained seeds — a different axis; we do not need that for a fixed-policy BoN/AWR Δ. See `METAWORLD_SINGLE_TASK_SPECIALIST.md`.

Paper targets (Table VI, per-task specialist):

| Task | Paper OAT₈ |
|------|------------|
| box-close | 44.4% |
| coffee-pull | 26.4% |
| disassemble | 17.2% |
| stick-pull | 9.6% |

### Data (regen 2026-07-10)
| Task | Zarr | Log |
|------|------|-----|
| box-close | `data/metaworld/box-close_N50.zarr` | `logs/metaworld_single_data_regen.log` |
| coffee-pull | `data/metaworld/coffee-pull_N50.zarr` | (same) |
| disassemble | `data/metaworld/disassemble_N50.zarr` | (same) |
| stick-pull | `data/metaworld/stick-pull_N50.zarr` | (same) |

Status: **DONE** — `[DONE] 2026-07-10T22:12:19` in regen log.

### Tokenizer (5001 epochs, DONE)
| Task | Run dir | Log | **Best ckpt (min mse)** | Best mse | Final mse @ep-5000 |
|------|---------|-----|-------------------------|----------|--------------------|
| box-close | `output/20260710/212943_train_oattok_mw-box-close_st_N50/` | `logs/train_oattok_mw-box-close_st_N50_s0.log` | `checkpoints/ep-3450_mse-0.019.ckpt` | **0.0186** | **0.0198** |
| coffee-pull | `output/20260710/212943_train_oattok_mw-coffee-pull_st_N50/` | `logs/train_oattok_mw-coffee-pull_st_N50_s0.log` | `checkpoints/ep-2670_mse-0.039.ckpt` | **0.0394** | **0.166** |
| disassemble | `output/20260710/235437_train_oattok_mw-disassemble_st_N50/` | `logs/train_oattok_mw-disassemble_st_N50_s0.log` | `checkpoints/ep-3410_mse-0.027.ckpt` | **0.0273** | **0.0284** |
| stick-pull | `output/20260710/235437_train_oattok_mw-stick-pull_st_N50/` | `logs/train_oattok_mw-stick-pull_st_N50_s0.log` | `checkpoints/ep-3030_mse-0.042.ckpt` | **0.0418** | **0.0472** |

> Final ≠ best: filenames round best mse; **Final** is `test_reconst_mse` at epoch 5000 from `logs.json`. Coffee-pull final **0.166** ≫ best **0.039** (late degradation — we freeze the min-mse ckpt).

Launch: `scripts/cluster_tokenizer_metaworld_single.sh`, `scripts/cluster_gen_metaworld_single_data.sh`

---

### MetaWorld single-task — coffee-pull

Old `coffee-pull` policy / chain5 / BoN / AWR artifacts were deleted on 2026-07-20 after checkpoint reset.

#### Policy refit — TopK locked (2026-07-21), train STOPPED

| Item | Value |
|------|-------|
| Run dir | `output/20260720/090816_train_oatpolicy_mw-coffee-pull_st_N50/` |
| Train seeds | `cfg.seed=0`, `training.seed=0` |
| Train eval | `n_test=250`, `rollout_every=200`, selection pool seed **1000**, topk k=3 |
| Tokenizer | `output/20260710/212943_train_oattok_mw-coffee-pull_st_N50/checkpoints/ep-2670_mse-0.039.ckpt` |
| TopK on disk | `ep-0600_sr-0.412` · `ep-0800_sr-0.420` · **`ep-1000_sr-0.432`** |
| Lock decision | post-ep1000 train-eval did **not** beat 0.432 → **Top-1 = ep-1000** |
| Lock file | `my_models/coffee_pull_topk_lock.txt` (`locked_at=2026-07-21T23:00:11+00:00`) |
| **Paper BASE_CKPT** | `output/20260720/090816_train_oatpolicy_mw-coffee-pull_st_N50/checkpoints/ep-1000_sr-0.432.ckpt` |
| Train stopped | `tmux coffee_policy_refit` killed after lock |

#### Matched (same MW protocol as stick/box/disassemble) — Wave1 RUNNING

| Item | Value |
|------|-------|
| Protocol | `test_start_seed=10000`, `n_test=50`, `-n 5`, OAT8, BoN N=8 vote; AWR collect `--seed 0` |
| Launch | `scripts/_launch_coffee_pull_matched_after_eval.sh` · tmux `coffee_pull_matched` |
| Wave1 out | `output/eval/matched_s10000/coffee-pull/{baseline_n5,bon_n8_n5,summary.json}` |
| Wave1 log | `logs/matched_s10000_coffee-pull_gpu1.log` |
| Wave2 (after Wave1) | `my_datasets/awr_s10000_coffee-pull.npz` · `my_models/awr_s10000_coffee-pull.ckpt` · `awr_n5/` |
| Status now | Wave1 **baseline** in progress; BoN/AWR not yet |

Anti-leak: train TopK SR **0.432 @ seed 1000** ≠ paper Table P (report only `matched_s10000`).

---

### MetaWorld single-task — stick-pull

#### Policy train — **STOPPED** (peak ep-800, degradation after, 2026-07-13)
| Item | Value |
|------|-------|
| tmux (was) | `mwst_pol_stick_pull` |
| GPU | 1 |
| Run dir | `output/20260711/134439_train_oatpolicy_mw-stick-pull_st_N50/` |
| Train log | `logs/train_oatpolicy_mw-stick-pull_st_N50_s0.log` |
| Metrics log | `output/20260711/134439_train_oatpolicy_mw-stick-pull_st_N50/logs.json` |
| Frozen tokenizer | `output/20260710/235437_train_oattok_mw-stick-pull_st_N50/checkpoints/ep-3030_mse-0.042.ckpt` |
| **Chain5 / BoN ckpt** | `.../checkpoints/ep-0800_sr-0.212.ckpt` |
| top-k on disk | `ep-0800_sr-0.212.ckpt`, `ep-1000_sr-0.200.ckpt`, `ep-1200_sr-0.188.ckpt` |

Train-time SR curve:

| Epoch | SR |
|-------|-----|
| 400 | 11.2% |
| 800 | **21.2%** (best) |
| 1000 | 20.0% |
| 1200 | 18.8% |
| 1400 | 14.8% |

Verdict: **peak @ ep-800**, monotonic decline after → stopped.

#### Eval — Chain5 — **DONE** (2026-07-13)
| Item | Value |
|------|-------|
| Script | `scripts/cluster_eval_mw_stick_pull_chain5.sh` |
| tmux (was) | `mwst_chain5_stick_pull` |
| Log | `logs/eval_mw_stick_pull_chain5.log` |
| Output dir | `output/eval/metaworld_stick-pull_paper5_ep0800/` |
| Summary | `output/eval/metaworld_stick-pull_paper5_ep0800/summary.json` |
| Per-seed dirs | `.../seed_{0..4}/` + `seed_{0..4}.log` |
| Base ckpt | `ep-0800_sr-0.212.ckpt` |
| **SR** | **16.4 ± 4.0%** |
| Per-environment-block | 8%, 14%, 30%, 20%, 10% |
| Paper target | 9.6% |

> Chain5 **below** train best (16.4% vs 21.2%) — high seed variance; train SR optimistic.

#### Eval — BoN → AWR pipeline — **DONE** (2026-07-14)
| Item | Value |
|------|-------|
| Script | `scripts/cluster_mw_stick_pull_bon_awr_pipeline.sh` + `scripts/resume_mw_stick_awr_step45.sh` |
| Logs | `logs/mw_stick_pull_bon_awr_pipeline.log`, `logs/mw_stick_pull_awr_resume.log` |
| Base ckpt | `ep-0800_sr-0.212.ckpt` |
| Chain5 baseline | **16.4 ± 4.0%** |

**BoN** — **DONE** (`n_test=50`, `-n 3`, vote; seeds `1000–1049`):
| Item | Value |
|------|-------|
| Output | `output/eval/metaworld_stick-pull_bon_n8_n3/` |
| **SR** | **30.7 ± 3.1%** (exp ≈ 0.34 / 0.30 / 0.28) |
| Δ vs chain5† | exploratory **+14.3 pp** |

**AWR** — **DONE** (train+eval after docker NVML restart):
| Item | Value |
|------|-------|
| Dataset | `my_datasets/awr_mw_stick_pull_bon.npz` |
| Collect stats | chunk-w SR 0.186; per-ep SR **0.237** / 1643 eps |
| AWR ckpt | `my_models/policy_awr_mw_stick_pull.ckpt` |
| Eval output | `output/eval/metaworld_stick-pull_awr_n3/eval_log.json` |
| **SR** | **39.3 ± 7.6%** |
| Δ vs chain5† | exploratory **+22.9 pp** (AWR > BoN here — strong distill on hard task; matched baseline still owed for paper Δ) |

---

### MetaWorld single-task — disassemble

#### Policy train — **STOPPED** (plateau then degradation, 2026-07-13)
| Item | Value |
|------|-------|
| tmux (was) | `mwst_pol_disassemble` |
| GPU | 0 |
| Run dir | `output/20260711/134440_train_oatpolicy_mw-disassemble_st_N50/` |
| Train log | `logs/train_oatpolicy_mw-disassemble_st_N50_s0.log` |
| Metrics log | `output/20260711/134440_train_oatpolicy_mw-disassemble_st_N50/logs.json` |
| Frozen tokenizer | `output/20260710/235437_train_oattok_mw-disassemble_st_N50/checkpoints/ep-3410_mse-0.027.ckpt` |
| **Chain5 ckpt** | `.../checkpoints/ep-1400_sr-0.700.ckpt` |
| top-k on disk | `ep-0800_sr-0.688.ckpt`, `ep-1200_sr-0.680.ckpt`, `ep-1400_sr-0.700.ckpt` |

Train-time SR curve:

| Epoch | SR |
|-------|-----|
| 600 | 65.2% |
| 800 | 68.8% |
| 1400 | **70.0%** (best) |
| 1600 | 67.2% |
| 1800 | 60.4% (degradation) |

Verdict: plateau ~68–70% for 800 ep, then drop at ep-1800 → stopped; chain5 uses **ep-1400**.

#### Eval — Chain5 — **DONE** (2026-07-14)
| Item | Value |
|------|-------|
| Script | `scripts/cluster_eval_mw_disassemble_chain5.sh` |
| tmux (was) | `mwst_chain5_disassemble` |
| Log | `logs/eval_mw_disassemble_chain5.log` |
| Output dir | `output/eval/metaworld_disassemble_paper5_ep1400/` |
| Summary | `output/eval/metaworld_disassemble_paper5_ep1400/summary.json` |
| Per-seed dirs | `.../seed_{0..4}/` + `seed_{0..4}.log` |
| Base ckpt | `ep-1400_sr-0.700.ckpt` |
| Train-time SR @ ckpt | 70.0% |
| **SR** | **66.4 ± 3.2%** |
| Per-environment-block | 66%, 64%, 56%, 72%, 74% |
| Paper target | 17.2% |

> Chain5 slightly below train best (66.4% vs 70.0%); still ≫ paper specialist 17.2%. Train SR mildly optimistic.

#### Eval — BoN → AWR — **TBD** (after chain5; not started)

---

### MetaWorld single-task — box-close

#### Policy train — **STOPPED on plateau** (2026-07-14)
| Item | Value |
|------|-------|
| tmux (was) | `mwst_pol_box_close` (killed) |
| GPU | 0 (freed for AWR resume) |
| Run dir | `output/20260711/134439_train_oatpolicy_mw-box-close_st_N50/` |
| Train log | `logs/train_oatpolicy_mw-box-close_st_N50_s0.log` |
| Metrics log | `output/20260711/134439_train_oatpolicy_mw-box-close_st_N50/logs.json` |
| Frozen tokenizer | `output/20260710/212943_train_oattok_mw-box-close_st_N50/checkpoints/ep-3450_mse-0.019.ckpt` |
| Settings | `lazy_eval=false`, `rollout_every=200`, `n_test=250`, `n_parallel_envs=4`, `checkpoint.topk.k=3` |

Train-time SR curve (`mean_success_rate`):

| Epoch | SR |
|-------|-----|
| 600 | 42.8% |
| 800 | 42.4% |
| 1000 | 42.0% |
| 1200 | 44.4% |
| 1400 | 45.6% |
| 1600 | 52.4% |
| 1800 | 52.8% |
| **2000** | **55.2%** (**best**) |
| 2200 | 50.4% |
| 2400 | 54.0% |
| ~2600 | train-only when stopped (no better top-k) |
| top-k on disk | `ep-1800_sr-0.528.ckpt`, `ep-2000_sr-0.552.ckpt`, `ep-2400_sr-0.540.ckpt` |

Verdict: **plateaued** — peak at ep-2000; next two evals below best (−4.8 / −1.2 pp). Stopped train; **matched / BoN / AWR ckpt = `ep-2000_sr-0.552.ckpt`**.

#### Eval — matched baseline — **RUNNING**; BoN / AWR — **TBD**

| Item | Path / status |
|------|---------------|
| Script | `scripts/cluster_eval_mw_box_close_matched_baseline.sh` |
| tmux | `mwst_matched_box_close` |
| Log | `logs/eval_mw_box_close_matched_baseline.log` |
| Output | `output/eval/matched/metaworld_box-close/baseline_n3/` |
| Protocol | `seed=1000`, `n_test=50`, `-n 3`, OAT8 |
| BoN / AWR | after baseline (same ckpt) — see Table B |
| Chain5 | not prioritized (sanity only) |

Paper target: **44.4%** (train-time 55.2% already above; not an independent final estimate).

---

### Active cluster tmux (2026-07-15 ~01:05 MSK)

| tmux | Task | Stage |
|------|------|-------|
| `mwst_matched_box_close` | box-close | **matched baseline RUNNING** (`seed=1000`, `n_test=50`, `-n 3`, ep-2000) |

Finished: coffee/stick AWR (`[DONE]` ~17:55 UTC); coffee/stick BoN; disassemble chain5; box train (plateau).


---

## Quick reference — all eval output dirs

```
output/eval/robomimic_lift_paper5_ep0600/
output/eval/robomimic_lift_paper5_ep1200/
output/eval/robomimic_lift_bon_n8_n3/
output/eval/robomimic_lift_awr_n3/

output/eval/robomimic_can_paper5_ep1700/
output/eval/robomimic_can_bon_n8_n3/
output/eval/robomimic_can_awr_n3/

output/eval/robomimic_square_paper5_ep0600/
output/eval/robomimic_square_paper5_ep1500/
output/eval/robomimic_square_bon_n8_n3/
output/eval/robomimic_square_awr_n3/

output/eval/metaworld_mt4_paper5_ep0450/
output/eval/metaworld_mt4_bon_n8_n3/
output/eval/metaworld_mt4_awr_n3/

# MetaWorld single-task specialists — chain5 (paper 250 eps)
output/eval/metaworld_coffee-pull_paper5_ep1000/     # DONE 43.2±3.3%
output/eval/metaworld_stick-pull_paper5_ep0800/      # DONE 16.4±4.0%
output/eval/metaworld_disassemble_paper5_ep1400/   # DONE 66.4±3.2%

# MetaWorld single-task — exploratory BoN / AWR (NOT Table B; Δ vs chain5 only)
output/eval/metaworld_coffee-pull_bon_n8_n3/         # DONE 28.0±0.0%
output/eval/metaworld_coffee-pull_awr_n3/           # DONE 29.3±2.3%
output/eval/metaworld_stick-pull_bon_n8_n3/         # DONE 30.7±3.1%
output/eval/metaworld_stick-pull_awr_n3/            # DONE 39.3±7.6%

# Matched ICRA dirs (Table B only)
output/eval/matched/metaworld_box-close/baseline_n3/  # RUNNING (first matched run)
# output/eval/metaworld_disassemble_bon_n8_n3/     # TBD
# output/eval/metaworld_disassemble_awr_n3/        # TBD
```

## Quick reference — AWR datasets & checkpoints

All four RoboMimic/MT4 AWR ckpts are on cluster under `my_models/` (Lift/Can restored 2026-07-13 from the original collect `.npz` with the same train hp, after the pipeline outputs were deleted from disk).

| Benchmark | AWR dataset | AWR policy ckpt |
|-----------|-------------|-----------------|
| Lift | `my_datasets/awr_lift_bon.npz` | `my_models/policy_awr_lift.ckpt` (464M) |
| Can | `my_datasets/awr_can_bon.npz` | `my_models/policy_awr_can.ckpt` (464M) |
| Square | `my_datasets/awr_s10000_square.npz` | `my_models/awr_s10000_square.ckpt` (matched 31.2±5.2%) |
| MT4 multitask | `my_datasets/awr_mt4_bon.npz` | `my_models/policy_awr_mt4.ckpt` (806M) |
| MW coffee-pull | rerun pending | rerun pending |
| MW stick-pull | `my_datasets/awr_mw_stick_pull_bon.npz` | `my_models/policy_awr_mw_stick_pull.ckpt` (39.3 ± 7.6%) |
| MW box-close | TBD | TBD |
| MW disassemble | TBD | TBD |

## Pipeline logs

| Pipeline | Log | Status |
|----------|-----|--------|
| Lift BoN→AWR | `logs/lift_bon_awr_pipeline.log` | DONE |
| Can BoN→AWR | `logs/can_bon_awr_pipeline.log` | DONE |
| Square BoN→AWR | `logs/awr_s10000_square_wave2_gpu0.log` · `..._eval_gpu0.log` | Wave2 **DONE** 2026-07-23 |
| MT4 BoN→AWR | `logs/mt4_bon_awr_pipeline.log` | DONE |
| MT4 data regen (single) | `logs/metaworld_single_data_regen.log` | DONE |
| MW coffee chain5 | `logs/eval_mw_coffee_pull_chain5.log` | DONE |
| MW stick chain5 | `logs/eval_mw_stick_pull_chain5.log` | DONE |
| MW disassemble chain5 | `logs/eval_mw_disassemble_chain5.log` | DONE |
| MW coffee BoN→AWR | rerun pending | old artifacts deleted |
| MW stick BoN→AWR | `logs/mw_stick_pull_bon_awr_pipeline.log` + `logs/mw_stick_pull_awr_resume.log` | DONE (AWR 39.3 ± 7.6%) |
| MW box-close policy train | `logs/train_oatpolicy_mw-box-close_st_N50_s0.log` | STOPPED plateau (best 55.2% @ep-2000) |
| MW box-close matched baseline | `logs/eval_mw_box_close_matched_baseline.log` | RUNNING |

---

*Last updated: 2026-07-15 ~01:05 MSK. coffee/stick AWR DONE; Table B (matched) added; box matched baseline running.*
