# OAT Reproduction — Results & Artifacts

Cluster container: `oat_mipt_robomimic_askhabaliev_gs`  
Workspace root: `/workspace/oat` (paths below are relative to `oat/`)

**Eval protocol notes:**
- **Chain5 (internal 250-episode estimate):** one trained checkpoint, 5 disjoint environment-seed blocks × 50 rollouts; `summary.json` in eval dir. This is **not** the paper's 5 independently trained seeds.
- **Train-time policy eval:** RoboMimic uses `n_test=50`, normally every 100 epochs; MT4 multitask used `n_test=50`, every 50 epochs; MetaWorld single-task specialists use `n_test=250`, every 200 epochs. Checkpoints are top-k by `mean_success_rate`.
- **BoN / AWR (quick pipeline):** `--n_test 50`, `--num_exp 3`, BoN `N=8 vote`; `eval_log.json` in eval dir.
- All use **OAT8:** `--use_k_tokens 8 --entropy_threshold 0`, `MUJOCO_GL=egl`.

**Cluster snapshot (2026-07-13 ~23:06 MSK):** RoboMimic + MT4 **DONE**. Active: **box-close** train mid ep-2000 eval (last logged SR **52.8%** @ep-1800); **disassemble chain5** seed **2/4** (done: seed0 **66%**, seed1 **64%**); **coffee/stick BoN** STEP1 Exp **2/3** (Exp1: coffee **28%**, stick **34%** — partial, not final).

**Status overview**

| Block | Status |
|-------|--------|
| RoboMimic (Lift / Can / Square) | **DONE** — chain5 + BoN/AWR pipelines complete |
| MetaWorld MT4 multitask | **DONE** — chain5 + BoN/AWR complete |
| MetaWorld single-task data + tokenizer | **DONE** |
| MetaWorld single-task policy train | **PARTIAL** — box-close **IN PROGRESS**; coffee / stick / disassemble **STOPPED** |
| MetaWorld single-task chain5 | coffee **DONE**, stick **DONE**, disassemble **IN PROGRESS**, box-close **TBD** |
| MetaWorld single-task BoN / AWR | coffee **IN PROGRESS**, stick **IN PROGRESS**, box / disassemble **TBD** |

---

## Summary table

| Benchmark | Chain5 baseline | BoN N=8 (3 exp) | AWR single (3 exp) | BoN Δ vs chain5† | AWR Δ vs chain5† |
|-----------|-----------------|-----------------|--------------------|---------------|---------------|
| **Lift** | **83.6 ± 1.7%** (ep-0600) | 91.3 ± 2.9% | **93.3 ± 1.8%** | +7.7 pp | +9.7 pp |
| **Can** | **86.0 ± 2.8%** (ep-1700) | 90.7 ± 1.3% | 89.3 ± 1.8% | +4.7 pp | +3.3 pp |
| **Square** (chain5 best) | **31.2 ± 1.5%** (ep-1500) | — | — | — | — |
| **Square** (BoN/AWR @ ep-0600) | **30.8 ± 2.7%** (ep-0600) | 38.0 ± 2.0% | **36.7 ± 2.9%** | **+7.2 pp** | **+5.9 pp** |
| **MT4 multitask** | **28.4 ± 3.1%** (ep-0450) | 26.7 ± 2.4% | 18.7 ± 1.8% | −1.7 pp | −9.7 pp |
| **MW coffee-pull** (specialist) | **43.2 ± 3.3%** (ep-1000) | *BoN Exp1=28%; Exp2/3 running* | *pending BoN* | — | — |
| **MW stick-pull** (specialist) | **16.4 ± 4.0%** (ep-0800) | *BoN Exp1=34%; Exp2/3 running* | *pending BoN* | — | — |
| **MW disassemble** (specialist) | *chain5 2/4* (so far 66%, 64%; train ep-1400 70.0%) | TBD | TBD | — | — |
| **MW box-close** (specialist) | *train ep-2000 eval*; last logged **52.8%** @ep-1800 | TBD | TBD | — | — |

**Square caveat:** BoN/AWR pipeline used **`ep-0600_sr-0.420.ckpt`**, not the better chain5 ckpt **`ep-1500`** (31.2%). Δ for BoN/AWR is vs **ep-0600 chain5 only** (30.8%). BoN/AWR eval is quick protocol (`n_test=50`, 3 exp), not chain5 250-eps — same as Lift/Can/MT4 pipelines.

† **Exploratory deltas, not the final matched estimator.** Chain5 averages 250 distinct environment initializations; quick BoN/AWR averages 3 stochastic runs on the same default 50 initializations (`1000–1049`). The arithmetic is correct, but a paper-ready causal comparison requires baseline, BoN, and AWR to use the same initialization set and repetition count.

Paper targets (Table VI, OAT₈): Lift **99.2%**, Can **80.8%**, Square **39.2%**, MT4 per-task specialists **44.4 / 26.4 / 17.2 / 9.6%**, MT4 avg **24.4%**.

---

## Publication-readiness audit (ICRA)

**Intended claim:** BoN and/or AWR improve **our own fixed OAT baseline**. Absolute agreement with OAT Table VI is a sanity check, not the main claim.

### Verified facts

- All completed SR values in the summary table match the cluster `summary.json` / `eval_log.json` artifacts.
- Baseline and BoN use the same base checkpoint within each pipeline; AWR is distilled from BoN rollouts generated from that checkpoint.
- AWR collection starts from environment seeds near 0 (parallel workers use distinct offsets), while final evaluation starts at 1000; no direct collection/eval seed overlap was observed in the completed RoboMimic datasets.
- RoboMimic data are the official multi-human image datasets converted to Zarr. MetaWorld single-task data are locally regenerated expert demonstrations.

### Limitations

- **Exploratory vs paper claim (RM/MW).** Summary-table Δ vs chain5 for RoboMimic / MetaWorld are **exploratory** until matched re-eval. Do **not** treat them as the same matched estimator used for LIBERO (baseline and BoN/AWR on one shared seed set, `-n 3`). For the paper: main text = matched only; chain5 mean = separate sanity-check, not the primary BoN/AWR comparison.
- **Comparison protocol / seed pairing.** Chain5 uses five disjoint environment-seed blocks (250 inits); quick BoN/AWR uses three stochastic repeats on seeds `1000–1049` (`n_test=50`). Uncertainty estimates are not comparable; sign/magnitude of Δ can flip (Can, MT4, coffee). Paper-ready: baseline, BoN, and AWR on one held-out init set and the same repetition count.
- **n_test / suite mismatch.** LIBERO selection results use `n_test=500`; RM/MW quick pipelines use `n_test=50`. Different eval regimes — do not mix suites into one causal claim without protocol notes (or rematch).
- **Can / MT4.** Until matched reruns exist, keep out of the central claim (or report as exploratory only). Chain5-mean Δ can over/under-state the paired effect.
- **Coffee (MetaWorld).** Treat as a **controlled negative** under matched evaluation (vote BoN can regress vs same-seed single-sample), not as a pipeline quirk or unexplained anomaly.
- **Fixed-policy vs Table VI.** Claim is improvement of a single OAT checkpoint under BoN/AWR, not five independently trained paper seeds. MT4 here is one shared four-task policy; Table VI specialists are a different protocol — not matched.
- **Checkpoint selection.** Some ckpts were picked on the same eval pool later used for reporting; absolute SR is a strong sanity check, not a fully held-out test. Paired Δ on a fixed ckpt is less affected. Square BoN/AWR used `ep-0600`, not best chain5 `ep-1500`.
- **Meta-World port.** Controlled demo port, not byte-for-byte original generator (original `sim-env` required ≥5 success timesteps then continued to 10; this port accepts on first success). Interpret MW numbers within that implementation.
- **Compute cost.** BoN raises SR via N candidates per replan; report latency/cost with SR, not accuracy alone.
- **Claim scope.** Adaptive-compute negatives and selection positives on LIBERO are the load-bearing story; RM/MW are generalization probes and stay subordinate until matched.

> RM/MW chain5↔quick-BoN deltas are exploratory. LIBERO remains the matched reference. Paper tables will use paired baseline/BoN/AWR on a common held-out seed set; chain5 stays a separate sanity estimate. Coffee is a controlled negative under matched eval, not an anomaly.

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

#### Eval — BoN → AWR pipeline — **IN PROGRESS** (2026-07-13)
| Item | Value |
|------|-------|
| Script | `scripts/cluster_mw_coffee_pull_bon_awr_pipeline.sh` |
| tmux | `mw_coffee_bon_awr` |
| Pipeline log | `logs/mw_coffee_pull_bon_awr_pipeline.log` |
| Base ckpt | `ep-1000_sr-0.432.ckpt` |
| Chain5 baseline | **43.2 ± 3.3%** |

**BoN** (expected paths):
| Item | Value |
|------|-------|
| Output | `output/eval/metaworld_coffee-pull_bon_n8_n3/` |
| Log artifact | `output/eval/metaworld_coffee-pull_bon_n8_n3/eval_log.json` |
| SR | TBD (STEP1: Exp1 **28%** done; Exp 2/3 running — not a final mean) |

**AWR** (expected paths):
| Item | Value |
|------|-------|
| Dataset | `my_datasets/awr_mw_coffee_pull_bon.npz` |
| AWR ckpt | `my_models/policy_awr_mw_coffee_pull.ckpt` |
| Eval output | `output/eval/metaworld_coffee-pull_awr_n3/` |
| Eval log | `output/eval/metaworld_coffee-pull_awr_n3/eval_log.json` |
| SR | TBD |

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

#### Eval — BoN → AWR pipeline — **IN PROGRESS** (2026-07-13)
| Item | Value |
|------|-------|
| Script | `scripts/cluster_mw_stick_pull_bon_awr_pipeline.sh` |
| tmux | `mw_stick_bon_awr` |
| Pipeline log | `logs/mw_stick_pull_bon_awr_pipeline.log` |
| Base ckpt | `ep-0800_sr-0.212.ckpt` |
| Chain5 baseline | **16.4 ± 4.0%** |

**BoN** (expected paths):
| Item | Value |
|------|-------|
| Output | `output/eval/metaworld_stick-pull_bon_n8_n3/` |
| Log artifact | `output/eval/metaworld_stick-pull_bon_n8_n3/eval_log.json` |
| SR | TBD (STEP1: Exp1 **34%** done; Exp 2/3 running — not a final mean) |

**AWR** (expected paths):
| Item | Value |
|------|-------|
| Dataset | `my_datasets/awr_mw_stick_pull_bon.npz` |
| AWR ckpt | `my_models/policy_awr_mw_stick_pull.ckpt` |
| Eval output | `output/eval/metaworld_stick-pull_awr_n3/` |
| Eval log | `output/eval/metaworld_stick-pull_awr_n3/eval_log.json` |
| SR | TBD |

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

#### Eval — Chain5 — **IN PROGRESS** (2026-07-13)
| Item | Value |
|------|-------|
| Script | `scripts/cluster_eval_mw_disassemble_chain5.sh` |
| tmux | `mwst_chain5_disassemble` |
| Log | `logs/eval_mw_disassemble_chain5.log` |
| Output dir | `output/eval/metaworld_disassemble_paper5_ep1400/` |
| Summary (when done) | `output/eval/metaworld_disassemble_paper5_ep1400/summary.json` |
| Per-seed dirs | `.../seed_{0..4}/` + `seed_{0..4}.log` |
| Base ckpt | `ep-1400_sr-0.700.ckpt` |
| Train-time SR @ ckpt | 70.0% |
| Paper target | 17.2% |
| SR | TBD (seeds 0–1 done: **66%**, **64%**; seed **2/4** running) |

> Train 70% vs paper 17.2% — chain5 is the authoritative number; train SR may be optimistic (top-k selected on same 250 test seeds).

#### Eval — BoN → AWR — **TBD** (after chain5)

---

### MetaWorld single-task — box-close

#### Policy train — **IN PROGRESS** (2026-07-13, possible plateau after ep-1800)
| Item | Value |
|------|-------|
| tmux | `mwst_pol_box_close` |
| GPU | 0 |
| Run dir | `output/20260711/134439_train_oatpolicy_mw-box-close_st_N50/` |
| Train log | `logs/train_oatpolicy_mw-box-close_st_N50_s0.log` |
| Metrics log | `output/20260711/134439_train_oatpolicy_mw-box-close_st_N50/logs.json` |
| Frozen tokenizer | `output/20260710/212943_train_oattok_mw-box-close_st_N50/checkpoints/ep-3450_mse-0.019.ckpt` |
| Launch | `scripts/cluster_launch_metaworld_single_policy_4.sh` |
| Script | `scripts/cluster_policy_metaworld_single.sh` |
| Settings | `lazy_eval=false`, `rollout_every=200`, `n_test=250`, `n_parallel_envs=4`, `checkpoint.topk.k=3` |

Train-time SR curve:

| Epoch | SR |
|-------|-----|
| 600 | 42.8% |
| 1200 | 44.4% |
| 1400 | 45.6% |
| 1600 | 52.4% |
| 1800 | **52.8%** |
| 2000 | *eval in progress* (chunk ~5/63) |
| top-k | `ep-1400_sr-0.456.ckpt`, `ep-1600_sr-0.524.ckpt`, `ep-1800_sr-0.528.ckpt` |

Verdict: large gain through ep-1600, then only +0.4 pp at ep-1800 → **possible plateau**; ep-2000 eval running — wait for it before selecting the chain5 checkpoint.

#### Eval — Chain5 / BoN / AWR — **TBD**

Expected artifact names (when launched):
| Item | Path pattern |
|------|--------------|
| Chain5 script | `scripts/cluster_eval_mw_box_close_chain5.sh` (TBD) |
| Chain5 output | `output/eval/metaworld_box-close_paper5_epXXXX/` |
| Chain5 log | `logs/eval_mw_box_close_chain5.log` |
| BoN/AWR script | `scripts/cluster_mw_box_close_bon_awr_pipeline.sh` (TBD) |
| BoN output | `output/eval/metaworld_box-close_bon_n8_n3/` |
| AWR dataset | `my_datasets/awr_mw_box_close_bon.npz` |
| AWR ckpt | `my_models/policy_awr_mw_box_close.ckpt` |
| AWR eval | `output/eval/metaworld_box-close_awr_n3/` |

Paper target: **44.4%** (already exceeded in train-time eval; this is not an independent final estimate).

---

### Active cluster tmux (2026-07-13)

| tmux | Task | Stage |
|------|------|-------|
| `mwst_pol_box_close` | box-close | policy train — ep-2000 eval |
| `mwst_chain5_disassemble` | disassemble | chain5 seed **2/4** (0=66%, 1=64%) |
| `mw_coffee_bon_awr` | coffee-pull | BoN STEP1 Exp **2/3** (Exp1=28%) |
| `mw_stick_bon_awr` | stick-pull | BoN STEP1 Exp **2/3** (Exp1=34%) |


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
output/eval/metaworld_disassemble_paper5_ep1400/   # IN PROGRESS

# MetaWorld single-task specialists — BoN / AWR (quick protocol)
output/eval/metaworld_coffee-pull_bon_n8_n3/         # IN PROGRESS
output/eval/metaworld_coffee-pull_awr_n3/           # TBD
output/eval/metaworld_stick-pull_bon_n8_n3/         # IN PROGRESS
output/eval/metaworld_stick-pull_awr_n3/            # TBD
# output/eval/metaworld_box-close_paper5_epXXXX/   # TBD after train plateau
# output/eval/metaworld_box-close_bon_n8_n3/       # TBD
# output/eval/metaworld_box-close_awr_n3/          # TBD
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
| MW coffee-pull | `my_datasets/awr_mw_coffee_pull_bon.npz` (TBD) | `my_models/policy_awr_mw_coffee_pull.ckpt` (TBD) |
| MW stick-pull | `my_datasets/awr_mw_stick_pull_bon.npz` (TBD) | `my_models/policy_awr_mw_stick_pull.ckpt` (TBD) |
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
| MW disassemble chain5 | `logs/eval_mw_disassemble_chain5.log` | IN PROGRESS |
| MW coffee BoN→AWR | `logs/mw_coffee_pull_bon_awr_pipeline.log` | IN PROGRESS |
| MW stick BoN→AWR | `logs/mw_stick_pull_bon_awr_pipeline.log` | IN PROGRESS |
| MW box-close policy train | `logs/train_oatpolicy_mw-box-close_st_N50_s0.log` | IN PROGRESS |

---

*Last updated: 2026-07-13 ~23:06 MSK. Re-sync after disassemble chain5 + coffee/stick BoN finishes.*
