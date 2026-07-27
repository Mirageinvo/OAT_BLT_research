# PAPER PLAN — writing brief for the LaTeX agent (AAAI-27)

Self-contained plan so a downstream agent can write the LaTeX paper independently.
Contains: thesis, structure decision, section-by-section plan, ALL current results/numbers,
figures/tables, related work, what to claim / NOT claim, and AAAI-27 format rules.
Companion files: `PAPER_MASTER.md` (fuller running notes), `abstract_aaai27.txt`,
`model_of_compounding.tex` (the math), `aaai27_reproducibility_checklist.tex`, `CLAUDE.md` (raw log).

STATUS: living document, under discussion. Sections marked [LOCKED] are decided; [OPEN] need a call.

---

## 0. Venue & format (AAAI-27) [LOCKED]
- `\documentclass[letterpaper]{article}` + `\usepackage[submission]{aaai2027}` (from AuthorKit27).
- **Anonymous double-blind:** author = "Anonymous Submission", empty affiliations, **anonymize
  self-references** (do not reveal it's the authors' prior work), no copyright footer on p.1.
- Two-column, letterpaper. Refs via `aaai2027.bst` (natbib) + `aaai2027.bib`.
- **FORBIDDEN** (paper rejected otherwise): any page break (`\newpage`/`\clearpage`/`\pagebreak`),
  `\pagestyle`, `\tiny`, `titlesec`, `fullpage`, `\columnsep`, negative `\vspace`/`\vskip` near any
  caption/figure/table/section/reference, `\nocopyright`.
- Title in **Mixed Case** (capitalize nouns/verbs/adjectives, both parts of hyphenated terms).
- **Page limit (AAAI-27 Main Technical Track) [CONFIRMED]:** **≤7 pages of MAIN CONTENT** — title,
  abstract, ALL text, figures, plots, tables, AND captions all count — **+ ≤2 pages for BIBLIOGRAPHY
  ONLY → total PDF ≤9 pages.** **Pages 8–9 may contain ONLY references** (NO experiments, tables,
  appendix, or conclusion there). **Target: 6.8–7.0 content pages + 1.5–2 reference pages.**
- **⛔ NO in-PDF appendix is possible** (pages 8–9 = references only). Any defensive/extra material
  goes to a SEPARATE supplementary file (if AAAI-27 permits one; the reproducibility checklist is
  separate). Reviewers may not be required to read it → **the 7-page body must be self-contained.**
- **7 pages is TIGHT** with the full figure set → must prioritize (see §6 Figures budget).

## 1. Title & thesis [LOCKED]
**Title:** *Spend Compute on Selection, Not Reduction: Rethinking Efficient Inference for
Action-Chunking Policies*
**Thesis:** For action-chunking policies, spending inference compute on **per-observation
reduction** (fewer action tokens K, or a longer replanning horizon R) does **not** beat a fixed
budget — closed-loop replanning **washes out** any single per-observation decision (shown by a
counterfactual oracle: per-state value(K), value(R) ≈ 0). Redirecting the **same** compute to
**candidate selection** (best-of-N over sampled chunks) **exceeds the base policy**, because a small
per-chunk gain **compounds** over the ~tens of replans in an episode; it distills into a single
forward pass (AWR). The prescription generalizes across LIBERO, robomimic, MetaWorld, and RoboCasa.

**POSITIONING [LOCKED 2026-07-27]: POSITIVE / method-led, NOT "adaptive inference fails."** The
headline is the POSITIVE result (selection exceeds the base policy — a claim no other paper can
counter-example). The reduction/adaptivity diagnosis is SUPPORTING (C3), and it is **scoped, never
universal**: we claim it for *long-horizon closed-loop chunk policies (LIBERO/OAT) under proper
controls (random baseline + counterfactual oracle)* — NOT "adaptive inference doesn't work." We
explicitly position as COMPLEMENTARY to adaptive-chunk/horizon methods (AAC, PACE, DEHP), which
report gains on other benchmarks (contact-rich), other signals, and other axes — do NOT contradict
them. Title kept ("Selection, Not Reduction") but the body must scope the negative + cite those
works as "working in other settings."

### Contributions (in order) [LOCKED]
- **C1 (headline — method + result):** verifier-free candidate SELECTION (best-of-N, mode-seeking
  consensus) **exceeds the base policy** on closed-loop chunk policies; cheap (one visual encoding
  amortized over N candidates); distillable to a SINGLE forward pass (AWR); generalizes across 4
  benchmarks (LIBERO, robomimic, MetaWorld, RoboCasa).
- **C2 (insight + model):** the gain is COMPOUNDING — a small per-chunk failure-probability
  reduction × ~tens of replans; formalized in a first-order model (ΔSR ∝ #replans), which unifies
  the positive (selection) and the null (reduction).
- **C3 (scoped diagnosis — supporting):** the same compounding explains why per-observation
  REDUCTION (adaptive K/R) does NOT help in this setting — a counterfactual oracle shows per-state
  value(K,R)≈0 (single decisions washed by replanning); learned/heuristic adaptive controllers ≈
  fixed/random. Scoped to long-horizon closed-loop; complementary to adaptive-chunk methods.

## 2. CORE STRUCTURE DECISION [LOCKED 2026-07-26]
**Mechanistic diagnosis (WHY reduction fails) is done on LIBERO-LONG ONLY**, because it needs the
counterfactual-sim harness (state save/restore) and a clean within-task compounding measurement.
Everything in the "Reduction fails" argument — oracle value(K,R)≈0, adaptive-K/R controllers ≈
random/fixed, compounding (within-task R-sweep + #replans + the model) — lives on LIBERO.
**The other three suites (robomimic, MetaWorld, RoboCasa) are used ONLY to verify the resulting prescription
generalizes: selection SR gains (BoN/AWR) + latency.**
- **#replans is measured only on LIBERO** (clean, via the within-task R-sweep). It is NOT used
  cross-suite: `mean_replans` is **confounded with SR** (failed episodes run to the horizon → more
  replans; e.g. robomimic Square 20.4 = failure-length, not "more decisions"). So do NOT plot a
  cross-suite ΔSR-vs-#replans regression.
- **Cross-suite is QUALITATIVE:** "the selection gain is larger on the long-horizon multi-stage
  LIBERO-LONG than on the shorter robomimic/MetaWorld" = *consistent with* compounding, not a proof.
- **State the division explicitly** in Setup, at each section transition, and in Limitations.
  Sentence: *"We conduct the counterfactual diagnosis on LIBERO; we then verify the resulting
  prescription (selection > reduction) generalizes on robomimic, MetaWorld, and RoboCasa."*
- Why this is fine (standard "analyze on one, generalize on others"): the oracle/branching study is
  expensive and harness-specific; the mechanism (compounding) is a general property of closed-loop
  chunk policies (argued via the model); the actionable POSITIVE prescription is verified multi-suite.

## 3. Section-by-section plan
Target ~7 pages. `[LIBERO]` = mechanism/proof; `[multi]` = generalization.

| § | Title | Content (what to argue) | Results to cite | Figs/Tables |
|---|-------|-------------------------|-----------------|-------------|
| 1 | **Introduction** (~1 pg) | Compute-allocation question for chunk policies. POSITIVE-led. Contributions in order: **C1** verifier-free selection **exceeds the base policy** (cheap via amortized vision, distillable to single-forward, multi-suite); **C2** a first-order **compounding** model (ΔSR ∝ #replans) unifying positive+null; **C3** scoped diagnosis (oracle: per-obs value(K,R)≈0 → reduction/adaptivity washed out on long-horizon closed-loop; complementary to AAC/PACE/DEHP, NOT contradicting them). | — | F1 teaser |
| 2 | **Setup** (~0.75 pg) | OAT: anytime, **prefix-decodable** action tokens (first k → a full chunk), **frozen** detokenizer; obs encoder + AR policy. Two axes: **K** (token depth, K∈[1,8]) and **R** (executed chunk length before replan). The **counterfactual-oracle** method (restore state → execute a choice → continue with full policy → measure true success). Benchmarks (see §Benchmarks below). Cost profile. | obs-enc 22.4M ≫ AR policy 5.0M (vision dominates) | — |
| 3 | **Reduction Does Not Pay** `[LIBERO]` (~1.5 pg) | (a) **Oracle: per-state value(K), value(R) ≈ 0** — the money shot; and reconstruction ≠ value. (b) **Adaptive controllers ≈ random/fixed at matched cost** (K: learned predictor ≈ obs-agnostic mixture, fixed k=4 dominates; R: convergence/PACE < random). (c) **Mechanism:** closed-loop replanning corrects any single-chunk choice → per-obs decisions wash out (bridge to §5). | oracle value_k **+0.043** (small, ~1.5σ; tracks fast-motion not contact); value_R **−0.028** (flat); corr(value,recon)≈**+0.08**. Fixed-R sweep **0.635/0.577/0.510/0.440** @R=8/16/24/32. convergence **0.553** < random **0.595** (matched R≈14). PACE **0.482** < random **0.537** < fixed **0.553** (matched R≈20). fixed k=4 **0.496** ≥ predictor/entropy. | F2 oracle≈0; F3 R-sweep + adaptive-R controls (all < random) |
| 4 | **Selection Pays** `[LIBERO headline + multi generalization]` (~1.75 pg) | (a) **Verifier-free best-of-N (vote)** exceeds base; **scaling law**; **vision amortized once** across N. (b) **AWR distillation** → single-forward, deployable. (c) **Generalization** on robomimic + MetaWorld + **RoboCasa** (the clean table; RoboCasa PENDING); gain tracks headroom & horizon (qualitative). (d) **Latency** (BoN ≈ flat wall-clock — vision amortized). | LIBERO base **0.581**→BoN **0.690/0.712/0.717** (sat ~0.72), AWR **0.684**. Multi-suite table (below): **7/7 Δ_BoN≥0 so far, agg +4.5pp; 7/7 Δ_AWR>0, agg +6.0pp** (→ 11 tasks when RoboCasa lands). | F4 BoN scaling (SR vs N); **Table 2 multi-suite** |
| 5 | **A Model of Compounding** `[LIBERO]` (~0.75 pg) | First-order model: per-step failure ρ; selection → ρ_N ≪ ρ; **ΔSR ≈ H(ρ−ρ_N)**, H = #replans. Prediction **ΔSR ∝ #replans**, confirmed by the **within-task R-sweep** (vary R on one LIBERO task → #replans changes, all else fixed → Δ_BoN grows as R shrinks). | per-chunk realizable headroom (isolate) **+0.024** is small; DEPLOYED BoN gives **+0.11** on LIBERO — the amplification is COMPOUNDING (the model), NOT literal ×34. In the model, ΔSR≈H(ρ−ρ_N) with a tiny per-step ρ−ρ_N≈0.003 × H≈34 → +0.11 (do NOT write '+0.024×34=+0.11'). ρ_N ≈ C(N,⌈N/2⌉)ρ^⌈N/2⌉. | F5 BoN×R within-task curve **[PENDING data — see §7]** |
| (supp) | **Does Learned Value Help? → SUPPLEMENTARY (NOT a numbered body section; body has 7 sections: 1 Intro / 2 Setup / 3 Reduction / 4 Selection / 5 Model / 6 Related / 7 Limitations+Conclusion)** | NO in-PDF appendix (pages 8–9 = refs only). Full defensive table goes to a separate supplementary file; the BODY carries only a 1–2 sentence self-contained mention. Content: chunk-Q / AWR+critic / IQL / signed-PG all ≈ or < simple selection → per-**chunk** value is noise; the gain is compounded **outlier-rejection**, not per-state value estimation. **BODY POINTER (§4/§5):** *"Learned value functions and offline-RL (chunk-Q, IQL, signed policy-gradient) do not beat simple consensus selection (supplementary)."* | value-BoN **0.476**; AWR+critic **0.652**; IQL **0.62**; signed-PG collapse **0.182**; all ≤ AWR **0.684**. | supp. table |
| 6 | **Related Work** (~0.5 pg) | see §Related below | — | — |
| 7 | **Limitations & Conclusion** (~0.5 pg) | single-arch (OAT); sim-only; **diagnosis LIBERO-centric, positive multi-suite**; RoboTwin dropped (out of regime). Conclusion: spend compute on selection (compounds), not per-obs reduction (washed out). | — | — |

### §1 Introduction — arc + hook [LOCKED 2026-07-27]
**Hook = B (positive-first). Name numbers up front. Frame GENERALLY ("action-chunking policies");
name OAT only in Setup.**

**Hook (draft — first two sentences, positive-led):**
*"Executing the consensus of a handful of sampled action chunks — a verifier-free best-of-N that
reuses a single perception pass — raises an action-chunking policy's success rate ABOVE its own base
policy (by 11 points on LIBERO-LONG), at negligible added cost. Spending the same compute the
'obvious' way — adapting the per-observation token budget or the replanning horizon — buys nothing.
Both follow from how per-chunk errors compound over an episode's many replanning steps."*

**6-beat arc:**
1. Setup + question (the hook): chunk policies replan periodically; spare inference compute → two
   ways: *reduce* per-obs (fewer tokens K / longer horizon R) vs *select* among candidate chunks.
2. The "obvious" reduction path + its intuition (anytime/adaptive compute). ONE sentence citing
   AAC/PACE/DEHP as "working on other benchmarks/axes" → scope + complementarity up front.
3. The finding (pivot): counterfactual oracle → on long-horizon closed-loop, per-obs value(K,R)≈0
   (replanning washes single decisions); the same compute in SELECTION exceeds the base policy.
4. Why (mechanism = compounding): a small per-chunk failure reduction compounds over ~tens of
   replans; the same compounding WASHES per-obs adaptivity and AMPLIFIES uniform selection
   (asymmetry: "where to spend" has no signal; "improve every decision a little" compounds).
5. What we do + payoff (NUMBERS here): verifier-free BoN (mode consensus) > base (+0.11 LIBERO),
   vision amortized once (negligible added latency), distills to a single forward pass (AWR),
   generalizes across 4 benchmarks / 2 simulators.
6. Contributions C1/C2/C3.

**Numbers to state in the intro:** +11 pts (LIBERO), "4 benchmarks / 2 simulators", "single forward
pass (AWR)", "negligible added cost (one perception pass amortized over N)".

**Framing precision (how we AVOID an overclaim rejection — MUST follow):**
- "Action-chunking policies" only for the QUESTION and the general MECHANISM (compounding is a
  property of the closed-loop chunk-policy CLASS). Name **OAT in Setup** as "our instantiation
  (a prefix-decodable action tokenizer + AR head)".
- Be precise about generality: R-axis, compounding, selection, amortized-vision are **generic** to
  encoder+AR chunk policies; the **K-axis reduction is OAT-specific** (needs prefix-decodable
  tokens) — do NOT claim K-generality. The cheap amortized-vision BoN is strongest on encoder+AR
  architectures (a wedge vs diffusion-VLA BoN like RoboMonkey), not universal.
- **Limitations MUST state:** single architecture (OAT); mechanism = class property, demonstration =
  OAT-only; future = Diffusion Policy / ACT / π0. Preempts "you overclaim generality."

### Benchmarks paragraph (in §2) [LOCKED content]
FOUR benchmarks across two simulator stacks (robosuite: LIBERO/robomimic/RoboCasa; MetaWorld), spanning episode horizon and base SR.
- **LIBERO-LONG (LIBERO-10)** — 10 long-horizon multi-stage tasks (single-arm Franka, robosuite),
  2 cameras + proprio, 7-DOF. **One multi-task policy** (task_uid); OAT ckpt `policy_ep-0250`
  (the filename SR 0.596 is the ckpt's train-time eval — do NOT report it; our clean baseline is 0.581, §5); eval 500 rollouts over 10 tasks, fresh init states. **Primary suite:
  longest horizon → strongest compounding; all diagnosis (oracle, controllers, R-sweep) here.**
- **robomimic (Lift, Can, Square)** — single-arm robosuite, human-teleop demos (~200/task),
  7-DOF, 2 cameras. **Per-task specialists** (retrained OAT). Spans difficulty: Lift ~90% (ceiling),
  Can ~76% (mid), Square ~30% (hard). Shorter horizon.
- **MetaWorld (coffee-pull, stick-pull, disassemble, box-close)** — MuJoCo/MetaWorld-v2, self-
  generated expert demos (50/task), 4-DOF. **Per-task specialists.** 2nd simulator; short horizon;
  difficulty 15–60%.
- **RoboCasa (close_drawer, coffee_press_button, turn_off_microwave, turn_off_sink_faucet)** —
  robosuite kitchen, **mobile manipulator (PandaOmron), Da=12** action; official 50 human + 150
  MimicGen demos (200/task). **Per-task specialists.** Protocol: literal-5 report seeds (10000–
  10004), n_test=50 each. Adds a distinct embodiment (mobile base + arm) and task family (kitchen).
  **[ALL RoboCasa NUMBERS PENDING — being RE-MEASURED fresh. IGNORE any prior RoboCasa SR figures
  (obsolete). If base lands below the OAT paper (0.54–0.64), the CLAIM stays Δ_BoN/Δ_AWR, NOT
  absolute Table-VI parity.]**
- **Honesty note (must be in text):** LIBERO = multi-task aggregate; robomimic/MW/RoboCasa = per-task
  specialists. Diagnosis = LIBERO-only; selection positive = multi-suite. Qualitative horizon
  ordering only (LIBERO-LONG longest), no cross-suite #replans regression.

### §3 — the counterfactual argument + TWO subtle points + anticipated objection [MUST be in the paper]

**The core measurement.** The counterfactual oracle is the only clean way to get the TRUE value of
a per-observation decision in closed loop (not a proxy). At a visited state s, save the sim state,
BRANCH: execute a choice (e.g. the chunk decoded at k=1 vs k=8; or execute R_small vs R_large
steps), then continue with the FULL policy to the episode end, measure true success. Average over
states (n=120, M=5 continuations). value(K)=p(k=8)−p(k=1); value(R)=p(R_s)−p(R_l).
Result: value_k ≈ +0.043 (small, ~1.5σ, tracks fast-motion not contact); value_R ≈ −0.028 (flat);
corr(value, reconstruction-gap) ≈ +0.08 → **reconstruction ≠ value**.
Why ≈0: after the chosen chunk executes, the policy REPLANS from the new state with the full budget,
so any single-chunk suboptimality is corrected. A single per-observation decision has ~zero value.

**SUBTLE POINT #1 (a reviewer WILL ask): "but the fixed-budget sweeps differ a lot!"** (fixed k=4
0.496 ≪ k=8 0.58; fixed R=8 0.635 ≫ R=32 0.44). Resolve explicitly: value(K) measures ONE swap
(one chunk different, the other ~33 full) → ≈0. Fixed k=4 = ALL ~34 chunks worse → a tiny per-chunk
effect × H replans → a large EPISODE difference. **The fixed-sweep gap is COMPOUNDING over all
chunks, not a per-state effect.** Consequence: there is no exploitable per-observation heterogeneity
(value flat across states/phases AND reconstruction doesn't predict it) → adaptive per-obs
allocation has nothing to condition on → controllers ≈ random/fixed (confirmed).

**SUBTLE POINT #2 (the main one): "if a single decision is ≈0, how does BoN — also per-chunk —
help?"** Both per-state values are small (value_k +0.043; BoN realizable +0.024) — the difference is
NOT magnitude. **BoN is not a per-state adaptive decision; it is a UNIFORM improvement to EVERY
chunk** (at each replan, pick the robust consensus among N diverse samples). Its small per-chunk
gain applies at all ~34 replans → compounds to +0.11. Adaptivity tries to decide WHERE to spend
(needs per-state heterogeneity → absent); selection improves EVERY decision a little (needs no
per-state signal). Same mechanism (compounding) WASHES OUT per-obs adaptivity (single decision ≈0)
and AMPLIFIES uniform selection (each decision slightly better × H).

**⭐ ANTICIPATED OBJECTION — "you could predict K and R adaptively at EVERY step and the effect would
also compound." MUST pre-empt this with a dedicated paragraph. Three-layer defense:**
1. **Empirical (the killer): we DID exactly this, and it compounds to NOTHING.** Learned per-step
   adaptive-K (token-count predictor, K per obs at every replan): lands on the fixed-k frontier ≈
   obs-agnostic mixture (w2.0 0.497@5.31, w4.0 0.559@6.29), fixed k=4 dominates. Per-step adaptive-R
   (convergence-signal, PACE): convergence 0.553 < random 0.595; PACE 0.482 < random 0.537. So
   per-step adaptive K and R, applied at every replan, compound to ≤ fixed/random. The objection's
   premise (a positive per-step gain to compound) is empirically FALSE.
2. **Mechanistic (why): compounding amplifies a per-step REDUCTION IN FAILURE PROBABILITY (ρ→ρ_N).**
   BoN provides one — it changes the ACTION (rejects a catastrophic outlier among diverse samples)
   → ρ_N < ρ. Adaptive-K/R changes the BUDGET but NOT the action (k2≈k8 reconstruction redundancy)
   and R is corrected by replanning → the oracle value(K,R)≈0 → there is no per-step failure
   reduction → 0 × H = 0.
3. **We measured the ORACLE upper bound, not a weak controller.** The single-deviation oracle
   value(K,R) is the best any adaptive controller could achieve — even it is ≈0 (and the +0.043
   residual tracks fast-motion = noise, and is NOT obs-conditionable: agnostic-mix = predictor,
   obs-wall). So it is not "our controller is bad"; there is no per-state signal to have.
- **Paper wording:** *"One might apply adaptive K/R at every replan and expect compounding. We test
  exactly this: learned and heuristic per-step controllers compound to ≤ fixed/random. The oracle
  explains why — per-step value(K,R)≈0 because the budget leaves the action (k2≈k8) and the
  replanning-corrected outcome unchanged, so there is no per-step gain to compound, unlike
  selection's outlier rejection."* This turns the dangerous objection into ANOTHER confirmation of
  the thesis (we anticipated AND tested it).

### Method note (for the agent — describe these precisely in §4)
- **Verifier-free best-of-N ("vote"):** per replan, encode vision ONCE → sample **N** candidate action
  chunks from the (cheap) AR head conditioned on the same visual features → pick the **mode-seeking
  consensus** in action space over the executed prefix (a KDE-density argmax = the candidate in the
  densest region — NOT the centroid, which averages multimodal plans into an invalid action) →
  execute it. This is variance reduction / catastrophic-outlier rejection. No trained verifier, no
  oracle. Cost: one perception pass amortized over N cheap AR rollouts.
- **AWR distillation (single-forward, deployable):** collect rollout chunks + the BoN-vote-selected
  tokens; **advantage-weighted supervised fine-tune of the AR head ONLY** (vision + tokenizer FROZEN)
  on those, with a KL-to-reference anchor (anti-collapse). Bakes the inference-time BoN gain into a
  SINGLE forward pass at deploy (no N-sampling at test time). Imitation-bounded (≈ the BoN source's
  per-episode SR minus the irreproducible inference-time selection part).

## 4. Multi-suite results (Table 2) — CLEAN, re-run 2026-07-26 (all ≥0, no negatives)
| suite | base | BoN-8 | BoN-16 | BoN-32 | AWR (1×) | Δ_BoN8 | Δ_AWR |
|---|---|---|---|---|---|---|---|
| Lift (robomimic) | 90.5±1.3 | 93.2±1.2 | 93.3±0.2 | 92.5±1.0 | 93.1±2.2 | +2.7 | +2.6 |
| Can (robomimic) | 76.3±2.4 | 85.6±2.2 | 83.1±1.3 | 84.8±1.6 | 83.2±3.1 | **+9.3** | **+6.9** |
| Square (robomimic) | 30.7±1.0 | 30.5±2.2 | 33.1±2.7 | 34.8±2.8 | 31.5±3.6 | −0.2→**+4.1@N32** | +0.8 |
| MW coffee-pull | 40.8±2.3 | 43.2±4.8 | — | — | 41.2±2.3 | +2.4 | +0.4 |
| MW stick-pull | 15.6±6.2 | 25.6±2.6 | — | — | 26.8±5.2 | **+10.0** | **+11.2** |
| MW disassemble | 62.4±5.2 | 63.2±6.3 | — | — | 69.6±5.7 | +0.8 | **+7.2** |
| MW box-close | 59.6±7.5 | 66.4±3.0 | — | — | 72.8±4.1 | **+6.8** | **+13.2** |
| **aggregate (7 so far)** | | | | | | **+4.5pp** | **+6.0pp** |
| RoboCasa ×4 (kitchen) | *PENDING* | *PENDING* | | | *PENDING* | *PENDING* | *PENDING* |
| **aggregate (11, when RoboCasa done)** | | | | | | *update* | *update* |
- BoN 7/7 ≥ 0 (Square only ~0 @N8, +4.1 by N=32 → hard-task-needs-more-N scaling). AWR 7/7 > 0,
  often > BoN on MW. Significant where headroom (Can/stick/box ~3σ); small tasks in-noise but +dir.
- The OLD Square −8.0 / Lift −3.6 were NOISE (re-run at higher power) → do NOT cite them.

## 5. LIBERO headline numbers (quick ref)
- OAT8 baseline **0.581 ± 0.012** (≈ paper 0.58). BoN vote: N=4 **0.662**, N=8 **0.690**, N=16
  **0.712**, N=32 **0.717** (saturates ~0.72; textbook log-N scaling). AWR-distill (single forward)
  **0.684** (~95% of BoN-8, at 1× inference).

## 6. Figures & Tables
- **F1 Teaser** (DONE — the current schematic): obs→OAT policy; "two ways to allocate inference
  compute" → Per-Observation Reduction (branches converge = washed out) vs Candidate Selection
  (branch→consensus→select, compounds, distill). TODO polish: add a tiny SR-vs-N inset + a light
  "washes out" vs "compounds, exceeds base" visual cue.
- **F2 Oracle value ≈ 0** (money shot): per-state value(K) and value(R) by phase/contact; overlay
  reconstruction-gap (corr≈0) → reconstruction ≠ value.
- **F3 R-sweep + adaptive-R controls:** SR vs R (8/16/24/32); overlay convergence, PACE, pace_raw,
  random at matched mean → all adaptive < random.
- **F4 BoN scaling law:** SR vs N (1/4/8/16/32), base dashed; log-N fit; saturate ~0.72.
- **F5 Compounding (within-task):** Δ_BoN vs R on one LIBERO task (BoN×R sweep) → gain grows as R
  shrinks (more replans). **[PENDING data: N=8 @ R∈{8,24,32}]**.
- **Table 1 Benchmarks:** suite / #tasks / embodiment / action-dim / demos / protocol / horizon.
- **Table 2 Multi-suite:** the clean table above.
- **Figure/table BUDGET (7 pages is TIGHT — prioritize):** MUST-HAVE in body = **F1 teaser, F2
  oracle≈0, F4 BoN scaling, F5 compounding (BoN×R), Table 2 multi-suite, + a compact latency table**.
  Compress or move to SUPPLEMENTARY if space is tight: F3 (R-sweep + adaptive-R controls — could fold
  into F2 or shrink), Table 1 (benchmarks — can be a text paragraph instead of a table).
- **SEPARATE SUPPLEMENTARY (not in-PDF):** K-axis Pareto (fixed{1,2,4,8}/predictor/entropy/agnostic-
  mix; fixed k=4 dominates); pow2 err(k); RL/value defensive table; per-task LIBERO-10 BoN [if
  measured]; full hyperparameters; reproducibility checklist.

## 7. PENDING experiments (before submission)
**REQUIRED (blocking — the paper needs all three):**
- **[REQUIRED] BoN×R sweep on LIBERO** — N=8 @ R∈{8,24,32} (have N=1 R-sweep + N=8@R16). The ONLY
  clean compounding evidence (§5, F5); Δ_BoN should grow as R shrinks. ~a few hours.
- **[REQUIRED] RoboCasa fresh numbers** — 4 tasks, base/BoN/AWR (colleague re-measuring). Fills
  Table 2 → aggregate over 11 tasks. All prior RoboCasa numbers obsolete.
- **[REQUIRED] Latency on ALL 4 benchmarks** — batch=1, base vs BoN-N (`measure_latency_adaptive.py`
  / equiv). Confirms **BoN adds ~flat wall-clock** (vision encoded once, amortized over N; AR is the
  cheap axis) → "negligible added latency" claim + per-benchmark **latency table (Table C)**. Median
  over ≥10 timed forwards, same GPU/docker as SR.

**nice-to-have (strengthen, not blocking):**
- **[nice-to-have] BoN-16/32 on MetaWorld** (match robomimic's scaling columns).
- **[nice-to-have] more seeds** on suggestive MW tasks (coffee/disassemble).
- **[optional] per-task LIBERO-10 BoN** (show BoN uniform across the 10, not 1–2 tasks).

## 8. Related work (cite; anonymize our own) — arxiv IDs inline for `\bibitem`s
- **Efficient VLA inference / token pruning:** VLA-Cache (2502.02175), LAC (2602.00686),
  VLA-Pruner (⚠ id not in our notes — agent to look up), LightVLA (⚠ look up).
- **Adaptive chunk length / horizon:** AAC (action-entropy chunk size, 2604.04161),
  PACE (speed-valley replan, 2606.00537), DEHP (2606.11408), AQC (2605.05544),
  StreamVLA (2602.01100). **Position as COMPLEMENTARY, not contradicted:** these report gains on
  other benchmarks (contact-rich), signals, and axes; we find that on long-horizon closed-loop
  LIBERO the per-observation value of such choices is ≈0 (oracle) and these controllers ≤ random/fixed
  under a matched-cost random control — a control PACE/DEHP omit. Do NOT claim "adaptive inference fails."
- **Best-of-N / verifiers for VLA:** RoboMonkey (sample+Gaussian+VLM-verifier, 2506.17811),
  MG-Select (verifier-free intrinsic-confidence, 2510.05681), RoVer (2510.10975),
  V-VLAPS (value-MLP MCTS, 2601.00969).
- **Ordered-token test-time search:** SoTo (image generation, 2604.15453, ICML 2026) — cite as the
  domain contrast (search helps generation, not closed-loop control).
- **RL / value for action chunks:** Q-chunking (2507.07969), V-GPS (2410.13816), VGAS (2602.07399),
  TGRPO (2506.08440), π_RL (2510.25889), CO-RFT (2508.02219), CRAFT (2605.04470).
- **Anytime / prefix-decodable action tokenizers:** OAT (our substrate — RSS 2026, cite the OAT paper;
  arxiv id ⚠ look up), FAST (2501.09747), FASTer (2512.04952), OmniSAT (2510.09667).
- **Real-time chunking / consistency:** RTC (2506.07339), BID (⚠ id not in our notes — look up).

**Arxiv-ID quick table (copy into `\bibitem`s; ⚠ = agent must look up):**

| paper | arxiv | paper | arxiv |
|---|---|---|---|
| VLA-Cache | 2502.02175 | SoTo | 2604.15453 |
| LAC | 2602.00686 | Q-chunking | 2507.07969 |
| VLA-Pruner | ⚠ | V-GPS | 2410.13816 |
| LightVLA | ⚠ | VGAS | 2602.07399 |
| AAC | 2604.04161 | TGRPO | 2506.08440 |
| PACE | 2606.00537 | π_RL | 2510.25889 |
| DEHP | 2606.11408 | CO-RFT | 2508.02219 |
| AQC | 2605.05544 | CRAFT | 2605.04470 |
| StreamVLA | 2602.01100 | FAST | 2501.09747 |
| RoboMonkey | 2506.17811 | FASTer | 2512.04952 |
| MG-Select | 2510.05681 | OmniSAT | 2510.09667 |
| RoVer | 2510.10975 | RTC | 2506.07339 |
| V-VLAPS | 2601.00969 | BID | ⚠ |
| OAT (RSS 2026) | ⚠ | | |

*Extra IDs from our notes if any of these get cited: SkiP 2605.15536, Spec-VLA 2507.22424,
AtomVLA 2603.08519, "Planning in 8 Tokens" 2603.05438, CF-VLA 2604.24622, MGP 2512.09101,
Discrete-Diffusion-VLA 2508.20072, RD-VLA 2602.07845, WorldPlanner 2511.03077,
Model-Predictive-Trees 2411.15651, RLinf-VLA 2510.06710, Memory-Consistent-NN 2310.06171.*

## 9. Limitations (state honestly)
- Single policy architecture (OAT); mechanism is a property of the closed-loop chunk-policy CLASS,
  demonstration is OAT-only. Future: Diffusion Policy, ACT, π0.
- Simulation only.
- **Diagnosis (oracle, controllers, compounding) is LIBERO-centric; selection positive is
  multi-suite.** Do not over-claim both across all suites.
- Compounding model is first-order (directional prediction ΔSR ∝ #replans, not exact numbers).
- **RoboCasa** (being re-measured; do NOT cite prior obsolete numbers): if base lands below the
  OAT paper (0.54–0.64), attribute to our training budget / single-task tokenizers (not the eval
  env), and keep the claim on Δ (BoN/AWR), not absolute Table-VI parity.
- **RoboTwin excluded** — bimanual + scripted narrow demos + absolute-joint → covariate shift,
  base ~3%, out of OAT's regime (large-pretrained-VLA territory). See PAPER_MASTER §8.

## 10. Decisions
- **[LOCKED] Positioning = POSITIVE / method-led** (C1 selection headline; C3 diagnosis is scoped
  supporting, complementary to adaptive-chunk methods — NOT "adaptive fails"). See §1.
- **[LOCKED] Title kept:** "Spend Compute on Selection, Not Reduction: ..." — body must scope the
  negative + cite AAC/PACE/DEHP as working in other settings.
- **[LOCKED] Contribution order:** C1 selection > C2 compounding > C3 scoped diagnosis.
- **[LOCKED] §6 RL/value → SEPARATE supplementary file + 1-sentence self-contained body pointer**
  (§4/§5). No in-PDF appendix is possible (AAAI-27 pages 8–9 = references only).
- **[LOCKED] RoboCasa = full 4th generalization suite** (on par with robomimic/MW). Numbers PENDING
  (colleague measuring: robosuite kitchen, mobile-manipulator PandaOmron, Da=12, 4 tasks close_drawer/
  coffee_press_button/turn_off_microwave/turn_off_sink_faucet, per-task specialists, literal-5 seeds).
  ALL numbers PENDING (being re-measured fresh — ignore any prior RoboCasa SR figures). CLAIM =
  Δ_BoN/Δ_AWR, NOT absolute Table-VI parity. Adds breadth (kitchen tasks, 12D mobile action).
- **[LOCKED] Venue = AAAI-27 Main Technical Track** — ≤7 content pages + ≤2 refs-only (≤9 total);
  pages 8–9 references ONLY. See §0.
- **[LOCKED] Timeline:** all 3 REQUIRED measurements (BoN×R / RoboCasa / latency) WILL be completed
  in time → the agent scaffolds their tables with placeholders now; numbers dropped in when ready.

## 11. HANDOFF to the writing agent

### Inputs the agent receives
- **This `PAPER_PLAN.md`** — the authoritative brief (structure, all current numbers, claims, format
  rules, what to claim / NOT claim).
- `abstract_aaai27.txt` (title + abstract v4), `model_of_compounding.tex` (the §5 math),
  `aaai27_reproducibility_checklist.tex`, AuthorKit27 (aaai2027.sty/.bst/.bib + template).
- **Figure 1 (teaser)** — provided SEPARATELY by the user (do not redraw; leave `\includegraphics`).

### Placeholder convention (for PENDING numbers)
Define once: `\newcommand{\PENDING}{\textcolor{red}{\textbf{[TBD]}}}`. Use `\PENDING` for every number
not yet measured, so they are greppable and obvious. **Do NOT invent or guess any number.** Cite only
the FINAL numbers below verbatim; everything else = `\PENDING`.

### FINAL numbers (use verbatim) vs PENDING (placeholder)
- **FINAL:** LIBERO base 0.581 / BoN 0.662(N4)/0.690(N8)/0.712(N16)/0.717(N32) / AWR 0.684; oracle
  value_k +0.043, value_R −0.028, corr(value,recon) +0.08; R-sweep 0.635/0.577/0.510/0.440; adaptive
  controls (convergence 0.553<random 0.595; PACE 0.482<0.537); K-axis (fixed k=4 0.496, predictor
  ≈agnostic-mix); defensive (value-BoN 0.476/AWR+critic 0.652/IQL 0.62/signed-PG 0.182); multi-suite
  robomimic (Lift/Can/Square) + MetaWorld (coffee/stick/disassemble/box) rows in Table 2.
- **PENDING (scaffold with `\PENDING`):** (1) RoboCasa 4-task rows in Table 2 + the aggregate over 11;
  (2) MetaWorld BoN-16/32 columns; (3) latency Table C (all 4 benchmarks, all cells); (4) F5 BoN×R
  points at R∈{8,24,32} (R=16 is real: Δ_BoN +0.113).

### Tables the agent MUST scaffold now (with placeholders)
- **Table 2 — Multi-suite** (in body): cols `Suite | Base | BoN-8 | BoN-16 | BoN-32 | AWR | ΔBoN8 |
  ΔAWR`. Fill Lift/Can/Square + 4×MW from §4 (MW BoN-16/32 = `\PENDING`); add 4 RoboCasa rows all
  `\PENDING`; aggregate row `\PENDING` (note "over 11 tasks when RoboCasa lands").
- **Table C — Latency** (compact, in body): cols `Benchmark | Base (ms) | BoN-8 | BoN-16 | BoN-32`,
  4 benchmark rows, ALL cells `\PENDING` (batch=1 median). Caption: "BoN adds ~flat wall-clock; one
  perception pass amortized over N."
- **Figure F5 — Compounding (BoN×R)**: plot Δ_BoN vs R; the R=16 point is real (+0.113), R∈{8,24,32}
  = `\PENDING` (leave the axis + a note). Caption states the prediction: Δ_BoN grows as R shrinks.
- **Table 1 — Benchmarks** (optional; may be a text paragraph to save space): suite / #tasks /
  embodiment / action-dim / demos / protocol / horizon.

### Instructions
- Follow §3 section order + budgets; POSITIVE/method-led (§1); scope the negative + cite AAC/PACE/DEHP
  as complementary (do NOT write "adaptive inference fails").
- **MUST include** the §3 anticipated-objection paragraph (adaptive-K/R-every-step → 3-layer defense).
- Anonymous submission (`\usepackage[submission]{aaai2027}`); anonymize self-references.
- Respect ALL forbidden commands (§0); NO page breaks; ≤7 content pages.
- §6 RL/value → separate supplementary + 1-sentence body pointer (no in-PDF appendix).
- Leave `\PENDING` everywhere a number is not yet available; produce a compile-clean skeleton.
