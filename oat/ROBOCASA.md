# RoboCasa — PAPER-STYLE protocol (canonical)

**Branch:** `robocasa` (from `robomimic`).  
**Status:** protocol locked **before** port. Code = TBD.  
**Claim:** matched Δ (BoN / AWR vs single-sample OAT8) on fixed ckpts — **not** Table VI absolute parity.

This file is the **only** paper protocol for RoboCasa. If a one-off script disagrees with this doc, the script is wrong.

---

## Why this exists (MetaWorld scars)

| MW mistake | RoboCasa lock |
|------------|---------------|
| Homemade demos + soft success (“success once”) | **Official HDF5 only**; success = upstream `_check_success` |
| Selection / report / collect seed bleed | **2000** / **10000** / **0** — never mixed |
| Exploratory `-n 3` / chain5 as paper Δ | Paper = **`matched_s10000`**, `-n 5` only |
| Reused exploratory AWR on new seeds | Wave 2 = **fresh** collect+train+eval |
| Latency as afterthought | Table C only after SR, paper_proof + same ckpts |

---

## 0. Non-negotiables

| Rule | Value |
|------|--------|
| Tasks | `close_drawer`, `coffee_press_button`, `turn_off_microwave`, `turn_off_sink_faucet` |
| Demos / task | **50 human + 150 machine = 200** ([OAT](https://arxiv.org/abs/2602.04215) App. A-A) |
| Action dim | **Da = 12** |
| Paper OUT | `output/eval/matched_s10000/robocasa/<task>/` |
| Report seeds | **`10000–10049`** for baseline, BoN, **and** AWR |
| Selection seeds | **`2000–2049`** TopK only |
| Collect seeds | **`0`** (+ worker offsets) |
| OAT8 | `--use_k_tokens 8 --entropy_threshold 0` |
| Sampling | `--temperature 1.0 --topk 10` |
| BoN | `--bon_free 8 --bon_signal vote` |
| Matched volume | `n_test=50`, `-n 5` (= 250 eps / method / task) |
| SR entrypoint | matched triplet script only (no ad-hoc paper evals) |

### Seed pools

| Pool | Seeds | Role | Paper? | Collect? |
|------|-------|------|--------|----------|
| selection | `2000–2049` | train-time TopK | ❌ | ❌ |
| report | `10000–10049` | Table P | ✅ | ❌ |
| collect | `0` | AWR data | — | ✅ |

---

## 1. Data — official only (G0)

**We do not generate RoboCasa demos for the paper path.**

| Action | Paper proof |
|--------|-------------|
| Download official RoboCasa / OAT-release **HDF5** (4 tasks) | App. A-A: 50H + 150M |
| Convert → `data/robocasa/<task>_N200.zarr` lossless | same keys as RM convert family: `action`, proprio, RGB cams |
| Assert `action.shape[-1] == 12` | App. A-A Da=12 |
| `validate_robocasa_data.py` green | count=200, NaNs, shapes |
| Log source URL/hash + zarr sha in `RESULTS_ROBOCASA.md` | reproducibility |

```bash
# Official RoboCasa v0.2 Box HDF5 only (human_im + mg_im). See RESULTS_ROBOCASA.md.
# Full mg_im ≈ 24GB/task → download one task at a time; slim to 150 MG demos (seed 0).
bash scripts/download_robocasa_paper_hdf5.sh CloseDrawer/human
bash scripts/g0_robocasa_one_task.sh close_drawer
# or chain all four (waits for each MG download):
# bash scripts/watch_g0_chain.sh close_drawer coffee_press_button turn_off_microwave turn_off_sink_faucet
python3 scripts/convert_robocasa_dataset.py --task close_drawer --seed 0   # 50H+150M → N200.zarr
python3 scripts/validate_robocasa_data.py
```

**Locked source:** RoboCasa tag `v0.2` `robocasa/utils/dataset_registry.py` (`human_im` / `mg_im`).  
**Not paper G0:** RoboCasa main LeRobot mirrors, Chaoqi HF H50+M500, or oracle regen.

| Do | Don't |
|----|--------|
| Official frozen demos | Oracle regen with custom accept rule |
| Upstream `_check_success` in env | “success_count > 0 once” |
| Full trajectories as recorded | Filter / truncate by softer success |
| N50 only for **debug** | Train paper tok/policy on N50 |

### If official HDF5 is unavailable

**Hard stop.** Escalate / find mirror.  
Regen is **not** the default backup. Only if leadership accepts: regen = Limitations + **G0b** parity vs reference env on same states — otherwise no paper RoboCasa.

### G0b — success-parity smoke (before policy)

1. Reference env = RoboCasa / OAT `sim-env` wrapper (pin versions).  
2. Replay K demos from our zarr open-loop.  
3. Per-step success must match reference `_check_success`.  
4. Log `logs/robocasa_success_parity_<task>.log` → PASS required.

---

## 2. Tokenizer — single-task (paper HP)

One frozen tok **per task**. Same tok for baseline / BoN / AWR on that task.

| Param | Lock (matches repo `train_oattok.yaml` / OAT) |
|-------|-----------------------------------------------|
| Encoder | 2 layers, emb 256, head_dim 64, **8 registers** |
| Decoder | 4 layers, emb 256, nested dropout `pow2` |
| FSQ | levels `[8,5,5,5]` |
| Ha / Hl | **32** / **8** |
| Data | actions from zarr (tok does not need RGB) |
| Train | batch **256**, epochs **5001**, seed **0**, top-3 by MSE |

```bash
uv run accelerate launch scripts/run_workspace.py \
  --config-name=train_oattok \
  task/tokenizer=robocasa/<task> \
  training.num_demo=200 \
  training.seed=0 \
  checkpoint.topk.k=3 \
  logging.mode=disabled
```

**Limitations wording (not “усиление”):**  
*We use task-specific tokenizers (as in our RM / MW-specialist track). Absolute Table VI is not the claim; matched Δ uses one tokenizer per task for all methods.*

---

## 3. Policy — selection ≠ report (G1)

| Stage | Seeds | Note |
|-------|-------|------|
| Train-time eval / TopK | `2000–2049` | never in Table P |
| Paper baseline / BoN / AWR | `10000–10049` | shared |

```bash
uv run accelerate launch scripts/run_workspace.py \
  --config-name=train_oatpolicy \
  task/policy=robocasa/<task> \
  task.policy.lazy_eval=false \
  task.policy.env_runner.n_test=50 \
  task.policy.env_runner.test_start_seed=2000 \
  policy.action_tokenizer.checkpoint=<TOK_CKPT> \
  training.num_demo=200 \
  training.seed=0 \
  training.rollout_every=100 \
  checkpoint.topk.k=3 \
  checkpoint.topk.monitor_key=mean_success_rate \
  logging.mode=disabled
```

Also lock in config: `policy_use_k_tokens=8`, `policy_entropy_threshold=0`, Da=12 normalizer.

**Pick ckpt:** max `mean_success_rate` on **2000–2049** only.  
**Do not** peek at 10000 until Wave 1. One BASE_CKPT for all later stages.

Policy HP (repo `train_oatpolicy.yaml`): batch 256, `policy_lr=5e-5`, `obs_enc_lr=1e-5`, epochs 5001, Ha=32, execute R=16 unless paper RoboCasa says otherwise — **freeze before first train** and write here if different.

---

## 4. Wave 1 — matched baseline + BoN (G2)

**Always fresh** on `10000–10049`. New folder only. Never “reuse old log as baseline.”

| Param | Value |
|-------|--------|
| `test_start_seed` | **10000** |
| `n_test` | 50 |
| `-n` | **5** |
| OAT8 | k=8, entropy=0 |
| BoN | N=8, `vote` |
| OUT | `output/eval/matched_s10000/robocasa/<task>/{baseline_n5,bon_n8_n5}/` |

```bash
SKIP_AWR=1 SUITE=<task> GPU=0 \
  BASE_CKPT=<ckpt_from_2000_topk> \
  TEST_START_SEED=10000 N_EXP=5 \
  bash scripts/cluster_matched_triplet.sh   # after RoboCasa suites wired
```

Raw equivalent (only if runner supports RoboCasa):

```bash
MUJOCO_GL=egl uv run scripts/eval_policy_sim.py \
  -c <BASE_CKPT> \
  -o output/eval/matched_s10000/robocasa/<task>/baseline_n5 \
  -n 5 --n_test 50 --test_start_seed 10000 \
  --use_k_tokens 8 --entropy_threshold 0 --temperature 1.0 --topk 10

# same + --bon_free 8 --bon_signal vote → bon_n8_n5/
```

**Δ_BoN** = BoN − this baseline (never vs TopK @2000).

### GATE → Wave 2

- [ ] `summary.json` + both eval_logs  
- [ ] `test_start_seed: 10000` inside summary  
- [ ] Table P draft updated  
- [ ] BoN useful → AWR; flat/anti → skip or report null  

---

## 5. Wave 2 — AWR from scratch (G3)

| Stage | Seeds | Why |
|-------|-------|-----|
| Collect | **0** | no leak into report |
| Train | **0** | reproducibility |
| Eval | **10000–10049** | same as baseline/BoN |

❌ Reuse exploratory / seed-1000 AWR. ❌ Collect on 10000 or 2000.

```bash
MUJOCO_GL=egl uv run python scripts/collect_awr_dataset.py \
  -c <BASE_CKPT> \
  -o my_datasets/awr_s10000_robocasa_<task>.npz \
  --seed 0 --bon_n 8 --n_chunks 20000 --n_workers 4

uv run python scripts/train_awr.py \
  -i my_datasets/awr_s10000_robocasa_<task>.npz \
  -c <BASE_CKPT> \
  -o my_models/awr_s10000_robocasa_<task>.ckpt \
  --beta 0.5 --beta_kl 0.05 --epochs 30

# eval → .../awr_n5/  (triplet SKIP_BASELINE_BON=1 or eval_policy_sim @10000)
```

**Δ_AWR** = AWR − Wave 1 paper baseline.

---

## 6. Latency — Table C / C′ (G4)

After SR only. Same ckpts as Table P.

```bash
# extend cluster_latency_paper_done.sh for robocasa suites
bash scripts/cluster_latency_paper_done.sh
FAIR_KV=1 bash scripts/cluster_latency_paper_done.sh   # Table C′
```

Required in `latency.json`: `paper_proof: true`, `git_commit`, paths to base + AWR, median over ≥10 timed forwards, batch=1, same docker/GPU as SR.

---

## 7. Anti-leak matrix (print before writeup)

| Stage | Seeds | Overlap with report? |
|-------|-------|----------------------|
| TopK | 2000–2049 | ❌ |
| Baseline / BoN / AWR eval | 10000–10049 | shared (OK) |
| AWR collect | 0 | ❌ |
| AWR train seed | 0 | ❌ |

```bash
rg -n "test_start_seed" output/eval/matched_s10000/robocasa/*/summary.json
# every hit must be 10000
```

---

## 8. Phase gates (no skip)

| Gate | Blocks | Pass |
|------|--------|------|
| **G0** | tokenizer | official 200, Da=12, validate OK |
| **G0b** | policy | success-parity PASS |
| **G1** | Wave 1 | TopK only on 2000; BASE_CKPT locked |
| **G2** | Wave 2 / cite BoN | baseline+BoN @10000 artifacts |
| **G3** | cite AWR | fresh collect@0 + eval@10000 |
| **G4** | Table C | paper_proof latency |

---

## 9. Table P (fill only from matched artifacts)

| Task | Base ckpt | Baseline | BoN N=8 | AWR | Δ_BoN | Δ_AWR | Lat med (S/B/A) |
|------|-----------|----------|---------|-----|-------|-------|-----------------|
| close_drawer | | | | | | | |
| coffee_press_button | | | | | | | |
| turn_off_microwave | | | | | | | |
| turn_off_sink_faucet | | | | | | | |

Forbidden: TopK@2000, exploratory n3, chain5-as-Δ, old AWR weights, any path outside `matched_s10000/robocasa/`.

---

## 10. Cluster order

1. Port env/runner/configs + convert/validate.  
2. **G0 + G0b** on `close_drawer`, then all 4 zarrs.  
3. 4× tokenizer → freeze MSE top-1.  
4. 4× policy (`test_start_seed=2000`) → lock BASE_CKPT.  
5. Wave 1 all tasks @10000 (`SKIP_AWR=1`).  
6. GATE → Wave 2 (collect 0 → train → eval 10000).  
7. Latency C (+ C′).  
8. Table P + Limitations (single-task tok; any forced port delta).

tmux: `rc_tok_<task>`, `rc_pol_<task>`, `rc_w1_<task>`, `rc_w2_<task>`.

---

## 11. Implementation backlog

| Component | Path |
|-----------|------|
| Env | `oat/oat/env/robocasa/` |
| Runner | `oat/oat/env_runner/robocasa_runner.py` (default selection seed **2000**) |
| Convert / validate / parity | `scripts/convert_robocasa_dataset.py`, `validate_robocasa_data.py`, `robocasa_success_parity.py` |
| Configs | `oat/config/task/{tokenizer,policy}/robocasa/*.yaml` |
| Matched / latency | extend `cluster_matched_triplet.sh`, `cluster_latency_paper_done.sh` |
| Results | `oat/RESULTS_ROBOCASA.md` |

---

## ✅ Done per task

- [ ] G0 official N200 + validate  
- [ ] G0b success-parity  
- [ ] Tok + policy; TopK @2000 locked  
- [ ] G2 Wave1 @10000 (triplet)  
- [ ] G3 Wave2 fresh AWR @10000 (or explicit skip)  
- [ ] G4 latency paper_proof  
- [ ] Table P + Limitations  

**First milestone:** official HDF5 → zarr + G0b on `close_drawer` — **before** any tokenizer train.
