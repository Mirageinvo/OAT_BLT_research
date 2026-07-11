# OAT Reproduction — Results & Artifacts

Cluster container: `oat_mipt_robomimic_askhabaliev_gs`  
Workspace root: `/workspace/oat` (paths below are relative to `oat/`)

**Eval protocol notes:**
- **Chain5 (paper-style):** 5 eval seeds × 50 rollouts = **250 episodes**; `summary.json` in eval dir.
- **BoN / AWR (quick pipeline):** `--n_test 50`, `--num_exp 3`, BoN `N=8 vote`; `eval_log.json` in eval dir.
- All use **OAT8:** `--use_k_tokens 8 --entropy_threshold 0`, `MUJOCO_GL=egl`.

Single-task MetaWorld specialists → **TBD** (section stub at bottom).

---

## Summary table

| Benchmark | Chain5 baseline | BoN N=8 (3 exp) | AWR single (3 exp) | BoN Δ vs base | AWR Δ vs base |
|-----------|-----------------|-----------------|--------------------|---------------|---------------|
| **Lift** | **83.6 ± 1.7%** | 91.3 ± 2.9% | **93.3 ± 1.8%** | +7.7 pp | +9.7 pp |
| **Can** | **86.0 ± 2.8%** | 90.7 ± 1.3% | 89.3 ± 1.8% | +4.7 pp | +3.3 pp |
| **Square** | **31.2 ± 1.5%** (ep1500) | 38.0 ± 2.0% | *in progress* | +6.8 pp | — |
| **MT4 multitask** | **28.4 ± 3.1%** | 26.7 ± 2.4% | 18.7 ± 1.8% | −1.7 pp | −9.7 pp |

Paper targets (Table VI, OAT₈): Lift **99.2%**, Can **80.8%**, Square **39.2%**, MT4 avg **24.4%**.

---

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
| Train log | `logs/train_tok_lift.log` (also early runs under `output/.../.hydra/`) |
| **Best ckpt (top-1 mse)** | `output/20260704/203215_train_oattok_lift_N200/checkpoints/ep-1970_mse-0.006.ckpt` |
| Config | `task/tokenizer=robomimic/lift`, `training.num_demo=200`, `num_epochs=5001`, top-3 by `test_reconst_mse` |

### Policy
| Item | Value |
|------|-------|
| Run dir (paper) | `output/20260706/163500_train_oatpolicy_lift_N200/` |
| Train log | `logs/train_policy_lift_paper_s42.log` |
| **Chain5 ckpt** | `output/20260706/163500_train_oatpolicy_lift_N200/checkpoints/ep-0600_sr-0.920.ckpt` |
| Alt ckpt (chain5 also run) | `.../ep-1200_sr-0.900.ckpt` |
| Frozen tokenizer | `ep-1970_mse-0.006.ckpt` (above) |

### Eval — Chain5 (250 eps)
| Item | Value |
|------|-------|
| Output dir | `output/eval/robomimic_lift_paper5_ep0600/` |
| Log | `logs/eval_lift_chain5.log` |
| Summary | `output/eval/robomimic_lift_paper5_ep0600/summary.json` |
| **SR** | **83.6 ± 1.7%** (per-seed: 0.84, 0.86, 0.78, 0.88, 0.82) |
| Paper target | 99.2 ± 0.5% |

Also: `output/eval/robomimic_lift_paper5_ep1200/` → **77.6 ± 3.1%** (`logs/eval_lift_chain5_ep1200.log`).

### Eval — BoN → AWR pipeline
| Item | Value |
|------|-------|
| Pipeline script | `scripts/cluster_lift_bon_awr_pipeline.sh` |
| Pipeline log | `logs/lift_bon_awr_pipeline.log` |
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
| **SR** | **93.3 ± 1.8%** (exp: 0.92, 0.92, 0.88) |

> Note: `my_models/policy_awr_lift.ckpt` was written by the pipeline; on disk only `policy_awr_square.ckpt` / `policy_awr_mt4.ckpt` remain full-size — re-export from pipeline if needed.

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
| **SR** | **86.0 ± 2.8%** (per-seed: 0.90, 0.86, 0.86, 0.92, 0.76) |
| Paper target | 80.8 ± 2.3% |

### Eval — BoN → AWR pipeline
| Item | Value |
|------|-------|
| Pipeline script | `scripts/cluster_can_bon_awr_resume.sh` (resume from STEP2) |
| Pipeline log | `logs/can_bon_awr_pipeline.log` |
| Base ckpt | `ep-1700_sr-0.940.ckpt` |

**BoN:** `output/eval/robomimic_can_bon_n8_n3/` → **90.7 ± 1.3%** (`eval_log.json`)

**AWR:** dataset `my_datasets/awr_can_bon.npz` (collect per-ep SR **88.7%**); ckpt path `my_models/policy_awr_can.ckpt`; eval `output/eval/robomimic_can_awr_n3/` → **89.3 ± 1.8%**

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

Per-seed (ep1500): 0.32, 0.26, 0.30, 0.34, 0.34. Paper target: **39.2 ± 2.4%**.

### Eval — BoN → AWR pipeline
| Item | Value |
|------|-------|
| Pipeline script | `scripts/cluster_square_bon_awr_pipeline.sh` |
| Pipeline log | `logs/square_bon_awr_pipeline.log` |
| tmux | `square_bon_awr` |
| Base ckpt | `ep-0600_sr-0.420.ckpt` |

**BoN:** `output/eval/robomimic_square_bon_n8_n3/` → **38.0 ± 2.0%** (exp: 0.40, 0.34, 0.40)

**AWR:** dataset `my_datasets/awr_square_bon.npz`; ckpt `my_models/policy_awr_square.ckpt` (485 MB); eval dir `output/eval/robomimic_square_awr_n3/` — **IN PROGRESS** (STEP5, Jul 11)

---

## MetaWorld — MT4 multitask

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
| **Best ckpt (top-1 mse)** | `output/20260707/124135_train_oattok_mw-mt4_N50/checkpoints/ep-4010_mse-0.024.ckpt` |

### Policy
| Item | Value |
|------|-------|
| Run dir | `output/20260708/032431_train_oatpolicy_mw-mt4_N50/` |
| Train logs | `logs/train_policy_mt4_paper_s0.log`, `logs/train_policy_mt4_paper_s42.log` |
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
- Overall: **26.7 ± 2.4%**
- Per-task: box-close 48.7%, coffee-pull 46.2%, disassemble 8.3%, stick-pull 0%

**AWR collect** — `my_datasets/awr_mt4_bon.npz`
- 20k chunks, 4 tasks; per-episode SR **16.6%** (weak BoN source vs LIBERO ~72%)
- Per-task collect SR: box 15.4%, coffee 14.7%, disassemble 3.4%, stick 0%

**AWR train** → `my_models/policy_awr_mt4.ckpt` (845 MB, 30 ep)

**AWR eval** — `output/eval/metaworld_mt4_awr_n3/eval_log.json`
- Overall: **18.7 ± 1.8%** (regression vs baseline)
- Per-task: box-close 41.0%, coffee-pull 30.8%, disassemble 0%, stick-pull 0%

**Verdict:** no SR gain on MT4 multitask; BoN flat/−2 pp, AWR −10 pp (expected with 16% BoN collect SR).

---

## MetaWorld — single-task specialists

**Status: IN PROGRESS** — data regen done; tokenizer training running. Full results block to be filled after policy + chain5 + BoN/AWR.

### Data (fresh gen, 2026-07-10)
| Task | Zarr | Regen log |
|------|------|-----------|
| box-close | `data/metaworld/box-close_N50.zarr` | `logs/metaworld_single_data_regen.log` |
| coffee-pull | `data/metaworld/coffee-pull_N50.zarr` | (same) |
| disassemble | `data/metaworld/disassemble_N50.zarr` | (same) |
| stick-pull | `data/metaworld/stick-pull_N50.zarr` | (same) |

### Tokenizer (partial — not all runs finished 5001 epochs)
| Task | Run dir | Log | Best ckpt so far | Status |
|------|---------|-----|------------------|--------|
| box-close | `output/20260710/212943_train_oattok_mw-box-close_st_N50/` | `logs/train_oattok_mw-box-close_st_N50_s0.log` | `ep-3450_mse-0.019.ckpt` | stopped ~ep3450 |
| coffee-pull | `output/20260710/212943_train_oattok_mw-coffee-pull_st_N50/` | `logs/train_oattok_mw-coffee-pull_st_N50_s0.log` | `ep-2670_mse-0.039.ckpt` | stopped ~ep2670 |
| disassemble | `output/20260710/235437_train_oattok_mw-disassemble_st_N50/` | `logs/train_oattok_mw-disassemble_st_N50_s0.log` | TBD | **running** (`mwst_tok_disassemble`) |
| stick-pull | `output/20260710/235437_train_oattok_mw-stick-pull_st_N50/` | `logs/train_oattok_mw-stick-pull_st_N50_s0.log` | TBD | **running** (`mwst_tok_stick_pull`) |

Launch: `scripts/cluster_tokenizer_metaworld_single.sh`, `scripts/cluster_gen_metaworld_single_data.sh`

Policy / chain5 / BoN / AWR: **TBD** — see `METAWORLD_SINGLE_TASK_SPECIALIST.md`.

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
output/eval/robomimic_square_awr_n3/          # in progress

output/eval/metaworld_mt4_paper5_ep0450/
output/eval/metaworld_mt4_bon_n8_n3/
output/eval/metaworld_mt4_awr_n3/
```

## Quick reference — AWR datasets & checkpoints

| Benchmark | AWR dataset | AWR policy ckpt |
|-----------|-------------|-----------------|
| Lift | `my_datasets/awr_lift_bon.npz` | `my_models/policy_awr_lift.ckpt` |
| Can | `my_datasets/awr_can_bon.npz` | `my_models/policy_awr_can.ckpt` |
| Square | `my_datasets/awr_square_bon.npz` | `my_models/policy_awr_square.ckpt` |
| MT4 | `my_datasets/awr_mt4_bon.npz` | `my_models/policy_awr_mt4.ckpt` |

## Pipeline logs

| Pipeline | Log |
|----------|-----|
| Lift BoN→AWR | `logs/lift_bon_awr_pipeline.log` |
| Can BoN→AWR | `logs/can_bon_awr_pipeline.log` |
| Square BoN→AWR | `logs/square_bon_awr_pipeline.log` |
| MT4 BoN→AWR | `logs/mt4_bon_awr_pipeline.log` |

---

*Last updated: 2026-07-11 (cluster snapshot). Re-sync after square AWR eval and single-task MT4 pipeline complete.*
