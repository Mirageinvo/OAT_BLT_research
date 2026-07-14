# Resolution plan — ICRA 2027 (RoboMimic / MetaWorld track)

**Goal (this track):** paper-ready **matched evaluation** of BoN/AWR on our fixed OAT checkpoints for **RoboMimic + MetaWorld**, so RM/MW numbers can enter the paper without protocol lies.  
**Primary fix:** comparison protocol + claims text — **not** retrain (unless matched runs force a selector/ckpt change on a suite we need in main text).

**Ownership**

| Owner | Scope |
|-------|--------|
| **This track (us)** | RoboMimic (Lift / Can / Square), MetaWorld MT4 + single-task specialists; matched eval scripts; RM/MW tables; coffee controlled-negative writeup; RM/MW latency |
| **Scientific lead (LIBERO)** | LIBERO load-bearing story, adaptive K/R nulls, BoN/AWR on LIBERO, suite breadth / DP / contact-rich as they assign |

LIBERO in §0 is **paper context / protocol example**, not our TODO. We mirror the **matched** estimator the LIBERO track used (`same seeds`, `same n_test`, `-n` stochastic repeats for baseline and BoN/AWR), with suite-native `n_test` (RM/MW matched table: **50**).

Related: [`RESULTS.md`](RESULTS.md) (artifacts + limitations). Executable recipe: Cursor plan `icra_matched_paper_plan` (keep in sync with this file).

---

## 0. Paper summary (сводка — context only; Owner: PI / full paper)

### Context

OAT = closed-loop chunk policy. Per replan: observe → action chunk → execute prefix → replan.  
Levers: **K** (action tokens), **R** (exec horizon before replan).  
Question: spend budget on **saving** compute (adaptive K/R) or on **selection** (BoN)?  
Metric: episode SR. LIBERO base OAT ≈ **0.581** (lead track).

### Story axis (whole paper)

1. **Compounding** — single-chunk budget choices wash out under closed-loop replan; fixed-budget gaps accumulate over the episode.  
2. **Adaptive K/R fails** at matched cost (oracle + random controls) on forgiving settings.  
3. **Wrong signals** — reconstruction / phase / uncertainty decouple from task value.  
4. **Selection** — BoN (+ AWR distill) can exceed the base policy by accumulating a small per-replan edge.

**Slogan:** spend compute on **selection**, not per-step **savings**.

LIBERO key numbers, adaptive nulls, DP / contact-rich breadth → **lead track**. Our job is to make RM/MW evidence **protocol-clean** under the same story (including honest negatives).

### Related RL refs

- AWR: https://arxiv.org/abs/1910.00177  
- Q-chunking: https://arxiv.org/abs/2507.07969  

---

## 1. Methodology (what was wrong; how we fix it)

### Problem

RM/MW BoN/AWR were compared to **chain5 mean** (5 disjoint env-seed blocks × 50 = 250 inits, `num_exp=1` per block), while BoN/AWR used **3 stochastic repeats on one seed block** (`test_start_seed=1000`, `n_test=50`). Different variance, episodes, and init sets → **exploratory only**, not a paper causal Δ. Sign/magnitude can flip (Can, MT4, coffee).

### Fix (no retrain)

Re-run **inference only** for baseline, BoN, and AWR on one **shared fixed init set**:

| Param | Value |
|-------|--------|
| `test_start_seed` | `1000` (episodes `1000 … 1000+n_test-1`) |
| `n_test` | `50` |
| `-n` / `num_exp` | `3` for **all three** methods |
| Policy | OAT8: `--use_k_tokens 8 --entropy_threshold 0` |
| BoN | `--bon_free 8 --bon_signal vote` |

**Terminology (paper-correct):**

- Say **shared fixed init set** / **matched-seed estimator** — **not** “held-out / unused seeds” (`1000–1049` is often the same pool as train-time ckpt selection).  
- Say **matched** — **not** classic per-episode paired test (unless we store and pair per-ep successes).

**Claim:** relative Δ of BoN/AWR vs **matched baseline** on a fixed OAT ckpt. Chain5 absolute SR = **sanity / appendix**, not the BoN comparator.

### Suite status after rematch (decision rules)

| Suite | Role |
|-------|------|
| Lift / Square | Likely main positives if matched Δ holds |
| Can / MT4 | Out of central claim until matched; then include or appendix |
| coffee-pull | **Controlled negative** if matched BoN ≲ baseline (vote / selection limit) — not a pipeline bug; do **not** claim “short episode ⇒ weak compounding” without analysis |
| stick-pull | Matched so any gain is clean |
| box / disassemble | Same matched recipe only if they enter the paper (disassemble chain5 **DONE** 66.4%; box train still in progress) |

---

## 2. Our checklist (RM / MetaWorld only)

### A. Protocol

1. Force `--n_test 50 --test_start_seed 1000` on baseline, BoN, **and** AWR (do not rely on Hydra defaults — MT4 yaml is 250).  
2. Δ = method − **matched baseline mean** only.  
3. Prefer clean dirs under `output/eval/matched/<suite>/`; reuse old BoN/AWR `eval_log.json` only after verifying `num_exp=3` and same init protocol.  
4. Do not mix RM/MW (`n_test=50`) with LIBERO (`n_test=500`) into one pooled causal claim.

### B. Text

1. Exploratory (current chain5↔quick Δ) vs paper (matched Δ) — never confuse.  
2. Coffee = controlled negative under matched eval (+ optional `medoid` ablation).  
3. Cost/latency next to RM/MW selection claims.  
4. Limitations stay aligned with this file + `RESULTS.md`.

### C. Execution order

```text
0. Docs freeze (exploratory labels) — done / keep sync
1. scripts/cluster_matched_triplet.sh
2. Wait for free GPU if coffee/stick AWR collect still running
3. Matched runs: Can → MT4 → coffee → stick → Lift → Square
   then box/disassemble BoN+matched when policies ready
4. Coffee writeup; gate Can/MT4 into main vs appendix
5. Matched summary table in RESULTS; chain5 appendix
6. Latency single vs BoN N=8 on RM/MW (verify CLI supports BoN)
```

**Out of scope here:** LIBERO re-eval, LIBERO ordering ablation, other LIBERO suites, Diffusion Policy, contact-rich PACE contrast — **PI / lead track**.

**Retrain only if:** a suite we need in main text has matched Δ broken by a bad AWR/BoN source or vote is anti-informative and we need another selector for the positive claim.

---

## 3. Short verdict

For **this track**, paper-ready = matched RM/MW eval + honest tables/claims (incl. coffee negative).  
Checkpoints stay. LIBERO numbers and breadth are not our build blockers.  
See Limitations in `RESULTS.md`.
