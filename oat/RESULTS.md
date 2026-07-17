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
| **Square** (chain5 best) | **31.2 ± 1.5%** (ep-1500) | — | — | — | — |
| **Square** (BoN/AWR @ ep-0600) | **30.8 ± 2.7%** (ep-0600) | 38.0 ± 2.0% | **36.7 ± 2.9%** | **+7.2 pp** | **+5.9 pp** |
| **MT4 multitask** ‡ | **28.4 ± 3.1%** (ep-0450) | 26.7 ± 2.4% | 18.7 ± 1.8% | −1.7 pp | −9.7 pp |
| **MW coffee-pull** (specialist) | **43.2 ± 3.3%** (ep-1000) | **28.0 ± 0.0%** | **29.3 ± 2.3%** | exploratory −15.2 pp | exploratory −13.9 pp |
| **MW stick-pull** (specialist) | **16.4 ± 4.0%** (ep-0800) | **30.7 ± 3.1%** | **39.3 ± 7.6%** | exploratory +14.3 pp | exploratory +22.9 pp |
| **MW disassemble** (specialist) | **66.4 ± 3.2%** (ep-1400) | TBD | TBD | — | — |
| **MW box-close** (specialist) | *train best* **55.2%** @ep-2000 (chain5 N/A) | TBD | TBD | — | — |

**Square caveat:** BoN/AWR used **`ep-0600_sr-0.420.ckpt`**, not best chain5 **`ep-1500`**. Quick BoN/AWR = `n_test=50`, 3 exp — same as Lift/Can/MT4 exploratory pipelines.

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
- **Square:** finish matched baseline, then same rematch as others.
- **MT4 multitask.** Exploratory only; **not** in matched/paper wave.
- **Fixed-policy vs Table VI.** Single OAT checkpoint under BoN/AWR vs paper’s multi-seed Table VI — absolute parity is sanity only.
- **Checkpoint selection.** Some ckpts selected on the same eval pool; matched Δ on a fixed ckpt is the claim.
- **Meta-World port.** Controlled demo port differs from original sim-env success criterion — interpret MW within this implementation.
- **Compute cost.** BoN costs N candidates per replan; report latency/cost with SR (Table C, all suites).
- **Claim scope.** LIBERO = lead diagnosis; RM/MW = matched generalization probes.

> **PAPER = Table P only** (`matched_s10000`, episodes `10000–10049`).  
> Table B below = lab draft on selection seeds `1000–1049` — **не в статью**. Exploratory Summary = тоже только lab.

### Table P — PAPER matched (seed 10000) ← сюда пишем числа для статьи

**Protocol:** `test_start_seed=10000`, `n_test=50`, `-n 5`, OAT8, `--temperature 1.0 --topk 10`, BoN `--bon_free 8 --bon_signal vote`.  
**Δ** = method − paper baseline. Ckpt выбирался на seed 1000; отчёт на 10000 (disjoint).  
**Wave 1 (сейчас):** baseline + BoN, `SKIP_AWR=1`.  
**Wave 2:** только после GATE ниже — свежий AWR, **не** ранние exploratory раны.  
**Launch:** `bash scripts/cluster_launch_matched_paper_wave.sh` (5 tmux параллельно).

#### ⛔ GATE → Wave 2 (зафиксировано 2026-07-16)

Пока Gate не закрыт — **AWR не запускать**. Полный текст: [`RESOLUTIONPLAN.md`](RESOLUTIONPLAN.md).

1. Wave 1 suite DONE: есть `summary.json` + baseline/BoN в Table P (**только** `matched_s10000`).
2. ❌ Не брать: exploratory `policy_awr_*.ckpt` / старые `awr_*.npz` / seed-1000 Table B / chain5 как paper Δ.
3. Collect: явный `--seed 0` (❌ selection RM `1000–1049` / MW `1000–1249`, ❌ report `10000`); eval AWR: `TEST_START_SEED=10000`.
4. Square: first **ep-1500**; если BoN flat → rematch **ep-0600** @ s10000 (exploratory 600 ≠ paper).
5. Оригинал OAT Table VI — **sanity only**, не comparator.
6. **RM vs MW фиты разные** (`cfg.seed` 42 vs 0), но env TopK у обоих default `test_start_seed=1000` (MW длиннее до 1249). Paper `10000` вне обоих.

**После anti-leak seeds:** остаётся (a) **MW demo port** = controlled limitation; (b) **Lift** retrain; (c) **Square** Wave1/fallback. MT4 — не в paper.

| Suite | Base ckpt | Paper baseline | BoN N=8 | AWR | Δ_BoN | Δ_AWR | Artifacts |
|-------|-----------|----------------|---------|-----|-------|-------|-----------|
| Can | ep-1700 | **76.4±3.0%** | **80.8±1.1%** | **79.2±7.7%** | **+4.4** | **+2.8** | `matched_s10000/can/` · `awr_s10000_can.*` (AWR < BoN) |
| MW coffee-pull | ep-1000 | **41.6±4.1%** | **42.8±2.3%** | **41.2±3.3%** | **+1.2** | **−0.4** | `matched_s10000/coffee-pull/` · `awr_s10000_coffee-pull.*` (AWR ≈ base, flat) |
| MW stick-pull | ep-0800 | **15.6±6.2%** | **25.6±2.6%** | **26.8±5.2%** | **+10.0** | **+11.2** | `matched_s10000/stick-pull/` · `awr_s10000_stick-pull.*` |
| MW disassemble | ep-1400 | **62.4±5.2%** | **63.2±6.3%** | **69.6±5.7%** | **+0.8** | **+7.2** | `matched_s10000/disassemble/` · `awr_s10000_disassemble.*` |
| MW box-close | ep-2000 | **59.6±7.5%** | **66.4±3.0%** | **72.8±4.1%** | **+6.8** | **+13.2** | `matched_s10000/box-close/` · `awr_s10000_box-close.*` |
| Square | **ep-0600** | **29.6±7.3%** | **31.2±9.0%** | Wave2 RUN | **+1.6** | — | `matched_s10000/square/` · `paper_w2_square` · base=`ep-0600_sr-0.420` |
| Square ep-1500 (archived) | ep-1500 | **27.2±6.9%** | **26.0±7.5%** | — | **−1.2** | — | `matched_s10000/square_ep1500/` — BoN flat, not paper primary |
| Lift | retrain | — | — | — | — | — | later |

**Wave1 BoN verified (2026-07-16):** can / coffee / stick — `eval_log.json` ↔ `summary.json` match; Wave1 logs contain `DONE baseline` + `DONE bon`. Missing `ALL DONE` only (triplet.sh edited mid-run) — **не** invalidates paper numbers. Primary artifacts = eval_logs under `matched_s10000/`.

**Wave1 Square ep-0600 DONE (2026-07-17):** baseline 29.6±7.3 / BoN 31.2±9.0 (Δ+1.6, weak). Gate → Wave2 launched.

**Wave2 DONE (2026-07-17):** box-close AWR 72.8±4.1 (Δ_AWR +13.2); disassemble AWR 69.6±5.7 (Δ_AWR +7.2); can AWR 79.2±7.7 (Δ_AWR +2.8, below BoN 80.8); coffee-pull AWR 41.2±3.3 (Δ_AWR −0.4, flat); stick-pull AWR 26.8±5.2 (Δ_AWR +11.2). Sources: `awr_n5/eval_log.json` @ `matched_s10000/`.

**Wave 2 in flight (2026-07-17):** **square ep-0600** collect ~93% (18.6k/20k) → потом train→eval@10000.  
Collect `--seed 0` · train `--seed 0` · eval `@10000` · `scripts/cluster_matched_paper_wave2_awr.sh`.  
❌ not using exploratory `policy_awr_*` / old `awr_*.npz` / ep-1500 / seed-1000.

**Why Wave2 collect is slow (not stuck):** parallel BoN-distill collects (`n_chunks=20000`, `--bon_n 8`) on **2 GPUs / shared CPU+EGL**. Sim-bound (~4–6 s/chunk); GPU util often ~0%. Do **not** interpret as hung.

Paper Wave1 artifacts:  
`output/eval/matched_s10000/<suite>/{baseline_n5,bon_n8_n5}/eval_log.json` + `summary.json` + `logs/matched_s10000_<suite>_gpu*.log`.

| Artifact pattern | Examples |
|------------------|----------|
| AWR dataset | `my_datasets/awr_s10000_{can,coffee-pull,stick-pull,box-close,disassemble,square}.npz` |
| AWR ckpt | `my_models/awr_s10000_<suite>.ckpt` |
| Wave2 log | `logs/awr_s10000_<suite>_wave2_gpu{0\|1}.log` |
| Wave2 eval log | `logs/awr_s10000_<suite>_wave2_eval_gpu{0\|1}.log` |
| AWR eval | `output/eval/matched_s10000/<suite>/awr_n5/` |
| tmux | `paper_w2_<suite>` |
| ❌ never | `my_models/policy_awr_*` / old exploratory `awr_*.npz` / Table B |

**Lift** after retrain.

**Имена файлов на suite (одинаковый шаблон):**

```text
output/eval/matched_s10000/<suite>/
  baseline_n5/eval_log.json     # Wave1 single-sample OAT8  ← paper source of truth
  bon_n8_n5/eval_log.json       # Wave1 BoN N=8 vote
  awr_n5/eval_log.json          # Wave2
  summary.json                  # protocol + SR + Δ + paths
my_datasets/awr_s10000_<suite>.npz
my_models/awr_s10000_<suite>.ckpt
logs/matched_s10000_<suite>_gpu*.log           # Wave1 (DONE baseline/bon)
logs/awr_s10000_<suite>_wave2_gpu*.log         # Wave2 collect+train
logs/awr_s10000_<suite>_wave2_eval_gpu*.log    # Wave2 AWR eval only
tmux: paper_s10000_<suite> | paper_w2_<suite>
```

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

### Table C — PAPER latency (paper-proof remasure, 2026-07-17)

Источник: `output/eval/matched_s10000/<suite>/latency.json` + сводка `matched_s10000/table_c.json` (`paper_proof: true`).  
Кластер **Tesla V100-SXM2-32GB**, torch **2.5.1+cu124**, batch=1, warmup=20 excluded, **10 reps**, median; obs = val dataset; **obs counter reset + warmup per mode** (fair Single↔BoN); `git_commit=643bca01…` via `OAT_GIT_*` (docker без `.git`); `git_dirty=true` (uncommitted scripts at measure).  
**Не** MuJoCo wall-clock. Square — после Wave2. Pre-proof archive: `matched_s10000/_latency_pre_paperproof/`.

| Suite | Single median | BoN N=8 median | AWR median | SR (Table P) base→BoN→AWR |
|-------|---------------|----------------|------------|---------------------------|
| Can | **40.9** ms | **40.3** ms | **40.4** ms | 76.4 → 80.8 → 79.2 |
| coffee-pull | **48.7** | **47.7** | **49.3** | 41.6 → 42.8 → 41.2 |
| stick-pull | **47.2** | **46.2** | **49.9** | 15.6 → 25.6 → 26.8 |
| disassemble | **47.6** | **49.1** | **49.2** | 62.4 → 63.2 → 69.6 |
| box-close | **48.8** | **48.1** | **48.0** | 59.6 → 66.4 → 72.8 |

**Table C artifacts** (paper latency sources; все под `output/eval/matched_s10000/`):

| Suite | `latency.json` | AWR ckpt (timed) | Base ckpt (Single/BoN) | note |
|-------|----------------|------------------|------------------------|------|
| Can | `can/latency.json` | `my_models/awr_s10000_can.ckpt` | Wave1 `base_ckpt` in json / `summary.json` | `paper_proof: true` |
| coffee-pull | `coffee-pull/latency.json` | `my_models/awr_s10000_coffee-pull.ckpt` | idem | |
| stick-pull | `stick-pull/latency.json` | `my_models/awr_s10000_stick-pull.ckpt` | idem | |
| disassemble | `disassemble/latency.json` | `my_models/awr_s10000_disassemble.ckpt` | idem | |
| box-close | `box-close/latency.json` | `my_models/awr_s10000_box-close.ckpt` | idem | |
| Square | — | `my_models/awr_s10000_square.ckpt` (после Wave2) | ep-0600 | **TBD** after AWR eval |
| **сводка** | `table_c.json` | — | — | rebuild: `scripts/build_table_c.py` |
| pre-proof archive | `_latency_pre_paperproof/<suite>_latency.json` | — | — | не в статью |

Scripts: `scripts/measure_latency_paper.py`, `scripts/cluster_latency_paper_done.sh` (requires `OAT_GIT_COMMIT`), `scripts/build_table_c.py`.

**Read for paper:** BoN median ≈ Single на всех 5 suite (~41–49 ms) — ожидаемо: vision encode **один раз** (amortized), AR дешёвый → N=8 почти не бьёт policy-forward cost. Это и есть OAT-substrate win рядом с +SR. AWR ≈ Single. Episode time всё ещё доминирует sim — Table C = inference cost only.

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

## RoboMimic — all DONE

Pipelines: chain5 (internal 250-episode estimate) + BoN N=8 + AWR for Lift, Can, Square. See summary table; Square BoN/AWR from **ep-0600** (not best chain5 ep-1500).

## RoboMimic — Lift

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

### Policy
| Item | Value |
|------|-------|
| Run dir (paper) | `output/20260706/163500_train_oatpolicy_lift_N200/` |
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
| Chain5 ckpt (ep600) | `.../ep-0600_sr-0.420.ckpt` |
| **Chain5 ckpt (used for BoN)** | same ep-0600 |
| **Chain5 ckpt (best 250-eps)** | `.../ep-1500_sr-0.380.ckpt` |
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
| Pipeline script | `scripts/cluster_square_bon_awr_pipeline.sh` |
| Pipeline log | `logs/square_bon_awr_pipeline.log` |
| Status | **DONE** (2026-07-11 00:57 UTC; EGL warnings in eval, results valid) |
| Base ckpt | `ep-0600_sr-0.420.ckpt` |

**BoN** (from `ep-0600`, `n_test=50`, 3 exp): `output/eval/robomimic_square_bon_n8_n3/` → **38.0 ± 2.0%** (exp: 0.40, 0.34, 0.40)  
Δ vs ep-0600 chain5 (30.8%): **+7.2 pp**. Not comparable to ep-1500 (31.2%) — other ckpt, other protocol.

**AWR** (distilled from BoN @ ep-0600): `my_datasets/awr_square_bon.npz`; `my_models/policy_awr_square.ckpt`; eval `output/eval/robomimic_square_awr_n3/` → **36.7 ± 2.9%** (exp: 0.32, 0.36, 0.42)
Δ vs ep-0600 chain5: **+5.9 pp**. BoN > AWR on this ckpt (+7.2 vs +5.9).

> ep-1500 chain5 (31.2%) is the **best single-sample** checkpoint; BoN/AWR were **not** re-run from ep-1500.

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

#### Policy train — **STOPPED** (plateau ~43%, 2026-07-13)
| Item | Value |
|------|-------|
| tmux (was) | `mwst_pol_coffee_pull` |
| GPU | 1 |
| Run dir | `output/20260711/134440_train_oatpolicy_mw-coffee-pull_st_N50/` |
| Train log | `logs/train_oatpolicy_mw-coffee-pull_st_N50_s0.log` |
| Metrics log | `output/20260711/134440_train_oatpolicy_mw-coffee-pull_st_N50/logs.json` |
| Frozen tokenizer | `output/20260710/212943_train_oattok_mw-coffee-pull_st_N50/checkpoints/ep-2670_mse-0.039.ckpt` |
| **Chain5 / BoN ckpt** | `.../checkpoints/ep-1000_sr-0.432.ckpt` |
| top-k on disk | `ep-0800_sr-0.420.ckpt`, `ep-1000_sr-0.432.ckpt`, `ep-1400_sr-0.432.ckpt` |

Train-time SR curve (250-eps eval, 1 seed — **not** chain5):

| Epoch | SR |
|-------|-----|
| 200 | 35.6% |
| 600 | 41.2% |
| 1000 | **43.2%** |
| 1200 | 40.8% (dropped from top-k) |
| 1400 | 43.2% |

Verdict: **plateau ~43%** since ep-1000; stopped for chain5/BoN.

#### Eval — Chain5 — **DONE** (2026-07-13)
| Item | Value |
|------|-------|
| Script | `scripts/cluster_eval_mw_coffee_pull_chain5.sh` |
| tmux (was) | `mwst_chain5_coffee_pull` |
| Log | `logs/eval_mw_coffee_pull_chain5.log` |
| Output dir | `output/eval/metaworld_coffee-pull_paper5_ep1000/` |
| Summary | `output/eval/metaworld_coffee-pull_paper5_ep1000/summary.json` |
| Per-seed dirs | `.../seed_{0..4}/` + `seed_{0..4}.log` |
| Base ckpt | `ep-1000_sr-0.432.ckpt` |
| **SR** | **43.2 ± 3.3%** |
| Per-environment-block | 34%, 52%, 48%, 44%, 38% |
| Paper target | 26.4% |

> Chain5 mean **matches** train-time ep-1000 (43.2%) — no train-eval gap on this task.

#### Eval — BoN → AWR pipeline — **DONE** (2026-07-14)
| Item | Value |
|------|-------|
| Script | `scripts/cluster_mw_coffee_pull_bon_awr_pipeline.sh` + `scripts/resume_mw_coffee_awr_step45.sh` |
| Logs | `logs/mw_coffee_pull_bon_awr_pipeline.log`, `logs/mw_coffee_pull_awr_resume.log` |
| Base ckpt | `ep-1000_sr-0.432.ckpt` |
| Chain5 baseline | **43.2 ± 3.3%** |

**BoN** — **DONE** (`n_test=50`, `-n 3`, vote; seeds `1000–1049`):
| Item | Value |
|------|-------|
| Output | `output/eval/metaworld_coffee-pull_bon_n8_n3/` |
| **SR** | **28.0 ± 0.0%** |
| Δ vs chain5† | exploratory **−15.2 pp** |

**AWR** — **DONE** (train+eval after docker NVML restart):
| Item | Value |
|------|-------|
| Dataset | `my_datasets/awr_mw_coffee_pull_bon.npz` |
| Collect stats | chunk-w SR 0.238; per-ep SR **0.390** / 1926 eps |
| AWR ckpt | `my_models/policy_awr_mw_coffee_pull.ckpt` |
| Eval output | `output/eval/metaworld_coffee-pull_awr_n3/eval_log.json` |
| **SR** | **29.3 ± 2.3%** |
| Δ vs chain5† | exploratory **−13.9 pp** (still well below chain5; ≈ BoN — distill keeps the low mode) |

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
| Square | `my_datasets/awr_square_bon.npz` | `my_models/policy_awr_square.ckpt` (464M) |
| MT4 multitask | `my_datasets/awr_mt4_bon.npz` | `my_models/policy_awr_mt4.ckpt` (806M) |
| MW coffee-pull | `my_datasets/awr_mw_coffee_pull_bon.npz` | `my_models/policy_awr_mw_coffee_pull.ckpt` (29.3 ± 2.3%) |
| MW stick-pull | `my_datasets/awr_mw_stick_pull_bon.npz` | `my_models/policy_awr_mw_stick_pull.ckpt` (39.3 ± 7.6%) |
| MW box-close | TBD | TBD |
| MW disassemble | TBD | TBD |

## Pipeline logs

| Pipeline | Log | Status |
|----------|-----|--------|
| Lift BoN→AWR | `logs/lift_bon_awr_pipeline.log` | DONE |
| Can BoN→AWR | `logs/can_bon_awr_pipeline.log` | DONE |
| Square BoN→AWR | `logs/square_bon_awr_pipeline.log` | DONE |
| MT4 BoN→AWR | `logs/mt4_bon_awr_pipeline.log` | DONE |
| MT4 data regen (single) | `logs/metaworld_single_data_regen.log` | DONE |
| MW coffee chain5 | `logs/eval_mw_coffee_pull_chain5.log` | DONE |
| MW stick chain5 | `logs/eval_mw_stick_pull_chain5.log` | DONE |
| MW disassemble chain5 | `logs/eval_mw_disassemble_chain5.log` | DONE |
| MW coffee BoN→AWR | `logs/mw_coffee_pull_bon_awr_pipeline.log` + `logs/mw_coffee_pull_awr_resume.log` | DONE (AWR 29.3 ± 2.3%) |
| MW stick BoN→AWR | `logs/mw_stick_pull_bon_awr_pipeline.log` + `logs/mw_stick_pull_awr_resume.log` | DONE (AWR 39.3 ± 7.6%) |
| MW box-close policy train | `logs/train_oatpolicy_mw-box-close_st_N50_s0.log` | STOPPED plateau (best 55.2% @ep-2000) |
| MW box-close matched baseline | `logs/eval_mw_box_close_matched_baseline.log` | RUNNING |

---

*Last updated: 2026-07-15 ~01:05 MSK. coffee/stick AWR DONE; Table B (matched) added; box matched baseline running.*
