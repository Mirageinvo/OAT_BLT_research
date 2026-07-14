# Resolution plan — ICRA 2027 paper-ready

**Goal:** main-conference ICRA 2027.  
**Primary fix now:** comparison protocol + text framing — **not** new training (unless matched reruns show a task/selector/ckpt that breaks the story).

Related: `RESULTS.md` (artifacts + limitations), this file (what to do).

---

## 0. Paper summary (сводка)

### Context

We study **OAT** — a closed-loop chunk policy for robotic manipulation (LIBERO baseline).  
Each replan: observe (cameras/state) → generate an action chunk (~16–32 steps) → execute a prefix → replan.

Two per-step compute levers:

- **K** — how many action tokens (chunk fidelity);
- **R** — how many steps to execute before the next replan.

**Question:** under a fixed inference budget, should the policy **save** compute on easy observations (smaller K / shorter R), or **select** among several candidates?

**Metric:** episode success rate (SR). LIBERO OAT base = **0.581**.

### Abstract (paper-ready)

Vision-language-action policies generate action chunks and replan continuously,
raising a natural efficiency question: can a policy spend less compute per
observation -- fewer action tokens, or a shorter replanning horizon -- where the
task is easy, without losing success? Using OAT, a prefix-decodable action
tokenizer on LIBERO, we measure the true per-observation value of each compute
choice via counterfactual simulation. We find that per-observation adaptive
computation does NOT beat a fixed budget at matched cost: an oracle that knows
the true outcome assigns near-zero per-state value to token count and horizon,
and learned or heuristic controllers do no better than spending the same average
budget at random. The cause is COMPOUNDING -- closed-loop replanning washes out
single-chunk decisions, so apparent fixed-budget gains accumulate over the
episode rather than localizing to states. We further show the intuitive signals
fail: reconstruction error, motion phase, and model uncertainty all decouple
from task value. Conversely, the same compounding makes SELECTION pay off --
best-of-N and its distillation into a single-forward policy EXCEED the base
policy. We reconcile these results with adaptive-horizon methods that succeed in
contact-rich manipulation, arguing the boundary is decision recoverability.

### Four load-bearing ideas

1. **Compounding (mechanism).** A single per-step budget choice is washed out by the next replan. Fixed-budget gaps accumulate over the **episode**, not at isolated states. Property of closed-loop chunk policies, not OAT alone.
2. **Adaptive K/R fails (negative).** Per-observation compute reduction does not beat fixed budget at matched cost; oracle per-state value of K and R ≈ 0; learned/heuristic controllers ≤ random spend of the same mean budget.
3. **Wrong signals.** Reconstruction, motion phase (contact), and model uncertainty decouple from task value.
4. **Selection succeeds (positive).** Same compounding makes best-of-N (and AWR distillation) **exceed** the base policy.

**Slogan:** spend compute on **selection** (accumulates), not per-step **savings** (washed out).

### Key numbers (LIBERO, load-bearing)

| | |
|--|--|
| Base OAT | **0.581** |
| Fixed k | k=4 → 0.496; k=8 → ~0.58 |
| K-predictor ≈ agnostic mix | 0.525 ≈ 0.510 |
| Fixed R | 8→0.635, 16→0.577, 24→0.510, 32→0.440 |
| Adaptive R @ matched mean | convergence / PACE / pace_raw all **≤ random ≤ fixed** |
| Oracle value(k), value(R) | value_k small (+0.043), tracks fast motion not task; value_R ≈ 0; recon ≁ value |
| BoN vote | N=1→4→8→16→32: **0.581 / 0.662 / 0.690 / 0.712 / 0.717** (plateau ~0.72) |
| AWR | N=8 source → **0.659** (+0.078); N=16 @100ep → **~0.684** (resolved; doc once had noisy 0.66–0.69) |
| Cost | vision encoder ~22.4M ≫ policy ~5M; BoN amortizes vision across N |

### Conclusions (for the paper)

1. Adaptive K/R on forgiving closed-loop (LIBERO) does not help at matched cost — oracle + random controls; cause = compounding.
2. Intuitive signals fail → prior heuristics fail.
3. Selection is the right lever (BoN +0.11; AWR distill +0.08).
4. Novelty = **diagnosis**; BoN/AWR are demonstrations, not the core contribution.
5. Scope: “adaptive fails” = forgiving / recoverable settings. Contact-rich / low-recoverability (e.g. PACE domain) is the intended contrast boundary — not a contradiction.

### Related RL refs (needed)

- Advantage-weighted regression: https://arxiv.org/abs/1910.00177  
- Q-chunking: https://arxiv.org/abs/2507.07969  

---

## 1. What is wrong today (RM / MetaWorld vs LIBERO)

Done wrong relative to the LIBERO matched protocol and the summary above:

| Issue | What happened | Paper consequence |
|-------|---------------|-------------------|
| **Comparison target** | RM/MW BoN/AWR Δ vs **chain5 mean** | Not the same estimator as LIBERO matched Δ |
| **Seed regime** | Chain5 = 5×50 inits; quick = 3× stochastic on `1000–1049` | Sign/magnitude can flip (Can, MT4, coffee) |
| **n_test** | LIBERO `500`; RM/MW quick often `50` | Suites not causally poolable |
| **Coffee** | Read as anomaly / odd 28%×3 | Must be **controlled negative** under matched eval |
| **Framing** | Table Δ look like paper claims | Must be labeled **exploratory** until rematched |

**What can stay:** trained checkpoints, BoN/AWR pipeline code, Lift as likely positive; chain5 as sanity-only.

**No retrain required** for paper-ready protocol fix — only eval + text, unless matched runs show a broken story that needs a new selector/ckpt.

---

## 2. Paper-ready checklist (do this)

### A. Protocol (mandatory)

1. Pick one held-out init set per suite (e.g. seeds `1000–1049` for RM/MW `n_test=50`, or enlarge to match paper if claimed).
2. Run **matched**:
   - baseline: single-sample, OAT8, `-n 3` (same as BoN repeats);
   - BoN: N=8 vote, `-n 3`;
   - AWR: single-sample on AWR ckpt, `-n 3`.
3. Main-table Δ = BoN/AWR − **matched baseline only**.
4. Keep chain5 mean in appendix / “absolute SR sanity” — **not** the BoN comparison anchor.
5. Document `n_test` per suite; never mix LIBERO and RM/MW into one causal sentence without a protocol note.

### B. Claims / text (mandatory)

1. One story axis: **compounding → adaptive fails; selection accumulates**.
2. Split **exploratory** (current RM/MW chain5↔quick) vs **paper claim** (LIBERO matched + rematched RM/MW).
3. Coffee → **controlled negative** (matched regression / vote failure mode), not quirk.
4. Can / MT4 → out of central claim until matched; or matched then include.
5. Limitations already updated in `RESULTS.md` — keep in sync with paper Limitations.
6. Report **cost/latency** with SR for selection (vision amortized vs N×AR).

### C. Per-block status

| Block | Action | Main text? |
|-------|--------|------------|
| **LIBERO** | Keep as load-bearing matched evidence; optional ordering ablation; AWR16 already ~0.684 | **Yes** |
| **Lift** | Prefer matched `-n 3` baseline confirm; likely stays strong positive | Yes once matched |
| **Square** | Matched on same ckpt used for BoN (`ep-0600`); do not vs `ep-1500` without re-run | Yes if matched |
| **Can** | **Must** matched baseline/BoN/AWR; else appendix only | Blocked |
| **MT4** | Matched re-eval/reframing; mean-based sign may flip | Blocked |
| **coffee** | Matched + controlled-negative writeup; optional medoid/temp ablation | Yes as **negative** |
| **stick** | Matched so the gain is clean | Yes once matched |
| **box / disassemble** | Same protocol if they enter the paper | Only if matched |

### D. Breadth for main conf (after protocol)

From the summary §5 — still needed for “phenomenon not one setup”:

1. Multi-suite LIBERO (spatial / object / goal) — selection + at least one adaptive null.
2. Second model family (e.g. Diffusion Policy): analog of K (denoising steps) + R + selection.
3. ≥1 contact-rich contrast: where adaptive horizon **helps**, with **random** matched-cost control (PACE-style), to pin the recoverability boundary.

Optional / reviewer-forced: learned verifier past BoN plateau ~0.72; sustained oracle for “save every step”.

---

## 3. Concrete execution order (minimal path)

```text
1. Freeze claims: LIBERO = main; RM/MW Δ = exploratory until step 2.
2. Matched re-eval scripts (no train):
     baseline -n 3  |  Bon -n 3 (reuse existing logs only if seeds/n_test match)
     same ckpt, same n_test, same test_start_seed
3. Rebuild one paper table: matched baseline | BoN | AWR | Δ | cost
4. Rewrite coffee as controlled negative; Can/MT4 only if matched.
5. Latency microbench (batch=1) for BoN vs single — append cost column.
6. Then breadth: LIBERO suites → DP probe → contact-rich contrast.
```

**Retrain only if:** matched shows AWR/BoN source too weak to support the selection story on a suite you need in main text, or vote is systematically anti-informative and you need another selector for the positive claim.

---

## 4. Short verdict

Paper-ready work is mostly **fix protocol + interpretation + one matched table**.  
Checkpoints stay. New training is conditional.  
LIBERO already matches the summary; RM/MW must be rematched or demoted to exploratory — see Limitations in `RESULTS.md`.
