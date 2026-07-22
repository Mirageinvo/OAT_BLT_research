# PAPER MASTER DOC — "Spend Compute on Selection, Not Reduction"
Handoff for the writing/figures agent. Everything we found, the math, the paper plan, the
figure list, and what is still running. Numbers are simulation success rates (SR), mean ± std
over N independent eval runs unless noted. STATUS tags: **[DONE]**, **[RUNNING]**, **[PENDING]**.
Last updated 2026-07-22.

================================================================================
## 0. ONE-LINE THESIS
================================================================================
For closed-loop **action-chunking** policies, adaptively **reducing** per-observation compute
(fewer action tokens K, or shorter replanning horizon R) does **not** beat a fixed budget —
closed-loop replanning **washes out** any single decision (compounding). The **same** compounding
makes **selection** pay off: best-of-N over sampled chunks **exceeds** the base policy and
**distills** into a single-forward policy at no added cost. Novelty = the **diagnosis + the
compounding model**; the methods (best-of-N, AWR) are borrowed and used as demonstrations.

TITLE: **Spend Compute on Selection, Not Reduction: Rethinking Efficient Inference for
Action-Chunking Policies** (see `abstract_aaai27.txt` for the final abstract v4 + alternatives).

TL;DR: adaptive per-observation compute doesn't help action-chunking policies (replanning washes
it out); selecting among candidate chunks does — and distills to a single forward pass.

================================================================================
## 1. SETUP / BACKGROUND (for the Setup section)
================================================================================
**Policy = OAT** (Ordered Action Tokenization, a prefix-decodable action tokenizer):
- Frozen VQ-VAE/FSQ **tokenizer**: encodes a 32-step action chunk into **K=8 ordered tokens**,
  codebook size 1000 (FSQ levels [8,5,5,5]). `token_dropout_mode='pow2'`, `num_registers=8`.
  Prefix-decodable: any prefix of k tokens decodes a full 32-step chunk (anytime fidelity).
- **Policy** = `FusedObservationEncoder` (RobomimicRgbEncoder CNN over 2×128×128 cameras +
  proprio MLP) → transformer → **autoregressive head** predicting the token indices.
- **Parameter split (drives the cost model):** obs/vision encoder **22.4M** ≫ action tokenizer
  5.8M, AR head **5.0M**. Vision dominates cost.
- **Two adaptive axes:** **K** = number of tokens generated (autoregressive depth / chunk
  fidelity, K∈[1,8]); **R** = executed chunk length before replanning (`n_action_steps`, exec R
  of the 32 decoded steps). K is OAT-specific (needs ordered tokenizer); R is generic to any
  chunking policy.

**Benchmarks (3 suites, 2 simulator stacks):**
- **LIBERO** (LIBERO-10 / libero10, long-horizon, robosuite/MuJoCo). Base policy
  `policy_ep-0250`, base SR ≈ 0.58 (paper) / 0.581 (our eval).
- **robomimic** (Can, Square, Lift; robosuite). Retrained OAT per task.
- **MetaWorld** (coffee-pull, stick-pull, disassemble, box-close; Meta-World/MuJoCo). Retrained.
- Episode length / #replans (R=16): LIBERO ≈ 35 replans; robomimic ≈ 25 (max, horizon 400);
  MetaWorld ≈ 13 (max, horizon 200). [these are upper bounds; exact means [PENDING] — add a
  `mean_replans` log line in the runner].

**Counterfactual oracle method** (`branch_value_k.py`): restore the exact sim state
(`get_sim_state`/`set_state`), execute a candidate choice open-loop, then continue with the
default policy to episode end, average success over M continuations → the **true** per-state
value of that choice (ground truth, not a proxy). This is our main measurement tool.

================================================================================
## 2. RESULTS — NEGATIVE: per-observation adaptivity does not beat fixed
================================================================================

### 2.1 K-axis (token depth) — CLOSED NEGATIVE  [DONE]
- **pow2 constraint:** the tokenizer is trained (nested dropout) only on k∈{1,2,4,8}; k=3,5,6,7
  are UNTRAINED budgets → degraded. Offline reconstruction err(k): k1 .158, k2 .138, k3 .131,
  k4 .129, k5 .141, k6 .142, k7 .133, k8 .124 (dips at trained 4, jumps at 5–6). ⇒ only
  {1,2,4,8} are valid fixed budgets; k=5/6 are invalid baselines.
- **Fixed budgets:** k=4 → **0.496 ± 0.025**; k=8 → **0.58**.
- **Learned/heuristic controllers ≈ fixed / agnostic:**
  - learned predictor w=4.0 → 0.559 ± 0.014 @ 6.29 tok; w=2.0 → 0.497 ± 0.017 @ 5.31 tok.
  - entropy-threshold → 0.501 ± 0.016 @ 5.68 tok.
  - **fixed k=4 (0.496 @ 4.0) Pareto-dominates** predictor w=2.0 and entropy (same SR, fewer tok).
- **obs-conditioning is worthless (the key control):** obs-**agnostic** budget mixture at the
  predictor's marginal → **0.510 ± 0.020** vs obs-**conditioned** 4-class predictor **0.525**
  @ same 6.13 tokens. Gap +0.015 within noise ⇒ conditioning on the observation adds nothing
  over the mixing rate. **This is the sustained-policy proof for K.**
- Predictor collapses to k∈{1,2,8} (avoids untrained budgets); "mean 6.29" is a mix, never
  actually 6.

### 2.2 R-axis (replan horizon) — CLOSED NEGATIVE  [DONE]
- **Fixed-R sweep (K=8 full budget):** R=8 → **0.635**; R=16 → 0.577 (OAT default ≈ paper 0.58);
  R=24 → 0.510; R=32 → 0.440. SR rises steeply as R shrinks (more replanning = more success)
  but doubling replans doubles the dominant vision cost → a real trade-off, not free.
- **Adaptive-R < random at matched mean cost (decisive):**
  - convergence signal (decode-k2-vs-k8 divergence): **0.553 ± 0.036** vs **random 0.595 ± 0.019**
    vs fixed-R interp ~0.592, at matched mean R ≈ 14. convergence LOSES to random (~1.8σ).
  - PACE (kinematic speed-valley, [2606.00537]): **0.482 ± 0.012** vs random 0.537 ± 0.026 vs
    fixed R=20 0.553 ± 0.026 (matched mean R ≈ 19–20). PACE is WORST.
  - pace_raw (raw EE-translation speed): **0.502 ± 0.034** < random 0.537. Both speed definitions
    lose → "you implemented PACE wrong" attack closed.
  - **random ≈ fixed** at matched mean ⇒ SR(R) ~linear locally ⇒ no Jensen room ⇒ any adaptive-R
    win must come from genuine positive obs↔R correlation; none of the signals have it.

### 2.3 Oracle: per-state value ≈ 0 (the MONEY SHOT)  [DONE]
- **Grid oracle, multi-task** (`oracle_grid_mt.npz`, n=120, 10 tasks, M=5, k 1-vs-8, R 8-vs-32):
  - **value_k = +0.043** (K=1 vs 8, max fidelity contrast) — small, ~1.5σ above 0.
  - **value_R = −0.028** ≈ 0 (R=8 vs 32) — flat across all phases.
  - **value_k tracks FAST-MOTION, not contact:** fast_eef +0.087 ≫ slow_eef −0.000; low_contact
    +0.055 > high_contact +0.023; grip_change +0.043 = no_grip +0.043 (no grasp effect).
    ⇒ token fidelity matters for fast motion, NOT task-critical grasp/contact.
  - **corr(value_k, reconstruction_gap) = +0.079 ≈ 0** ⇒ reconstruction does NOT predict value.
- Interpretation: even an oracle with the true outcome assigns near-zero **exploitable** per-state
  value to K and R. Not blaming a weak controller — the signal itself is ~empty.

### 2.4 Supporting diagnoses  [DONE]
- **reconstruction ≠ value:** k2≈k8 in reconstruction (err .138→.124) but SR(k4→k8)=+0.08;
  value(k) tracks fast-motion not contact; a task-weighted tokenizer distortion study
  (`per_timestep_recon_error.py`) found grasp regions are reconstructed AS WELL AS average
  (steepness of gripper-weighted distortion = a discontinuity/sharpness artifact, not
  task-relevance) ⇒ a value-ordered tokenizer has nothing to reallocate.
- **value ≠ phase:** per-state value is flat across gripper-change / contact / velocity strata.
- **criticality ⊥ uncertainty:** GATE A (n=120) — candidate-plan **disagreement** ANTI-detects
  fate-deciding states (AUC(edge vs doomed)=0.316 single-task; ~0.5 multi-task) ⇒ decision
  criticality is decoupled from every confidence/uncertainty signal (obs, physics, generation
  disagreement, energy-OOD).
- **edge-concentration was a SINGLE-TASK ARTIFACT** (collection bug; fixed). Multi-task: no
  inverted-U, criticality flat across recoverability. (Report as a cautionary/negative; the
  edge-targeting method lead is CLOSED.)

================================================================================
## 3. RESULTS — POSITIVE: selection exceeds the base policy
================================================================================

### 3.1 Verifier-free best-of-N (inference-time)  [DONE]
Per replan: encode vision ONCE (amortized), sample N candidate chunks from the cheap AR head,
execute the **consensus** (mode-seeking KDE `vote`). K=8 full budget throughout.
**Scaling law (vs base 0.581 ± 0.012):**
| N | 1 (base) | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| SR | 0.581 ± 0.012 | 0.662 ± 0.025 | 0.690 ± 0.008 | 0.712 ± 0.012 | 0.717 ± 0.026 |
| Δ | — | +0.081 | +0.109 | +0.131 | +0.136 |
- Monotone, **decelerating, PLATEAU ≈ 0.72 reached at N=16** (N=32 within noise → operating
  point N=16). Textbook `SR ≈ base + a·log N`.
- **Selection EXCEEDS the base policy** (adaptivity never could — it's capped at base SR).
- Mechanism = mode-seeking / rejecting the rare catastrophic outlier (see Model §5).

### 3.2 AWR distillation (deployable, single-forward)  [DONE]
Collect rollouts WITH BoN-vote selection, log the selected chunk, label by episode success;
advantage-weighted SFT of the AR head only (vision+tokenizer frozen) + KL-to-ref anti-collapse.
- AWR, N=8 source, 30 ep → **0.659 ± 0.006**.
- AWR, N=16 source, 100 ep → **0.684 ± 0.027 (n=7)**  ← best deployable, **+0.103 over base**,
  ≈ inference-BoN N=8 (0.690) at 1× inference cost.
- **Imitation ceiling ≈ 0.68:** distill lands single-sample ~0.04–0.05 below the source's
  per-episode BoN SR; epochs/source don't close the gap.

### 3.3 Multi-suite generalization (robomimic + MetaWorld)  [DONE, more seeds PENDING]
BoN N=8 (vote) + AWR-distill vs base, matched settings.
| suite | base | BoN N=8 | AWR | Δ_BoN | Δ_AWR |
|---|---|---|---|---|---|
| Can (robomimic) | 76.4±3.0 | 80.8±1.1 | 79.2±7.7 | +4.4 | +2.8 |
| MW coffee-pull | 41.6±4.1 | 42.8±2.3 | 41.2±3.3 | +1.2 (flat) | −0.4 (flat) |
| MW stick-pull | 15.6±6.2 | 25.6±2.6 | 26.8±5.2 | **+10.0** | **+11.2** |
| MW disassemble | 62.4±5.2 | 63.2±6.3 | 69.6±5.7 | +0.8 (flat) | +7.2 |
| MW box-close | 59.6±7.5 | 66.4±3.0 | 72.8±4.1 | +6.8 | **+13.2** |
| Square (robomimic) | 29.6±7.3 | 31.2±9.0 | 24.8±2.3 | +1.6 (flat) | **−4.8** (AWR regresses) |
| **aggregate (6)** | | | | **+4.1pp** | **+4.9pp** |
- **BoN = robust positive-or-neutral** (6/6 Δ_BoN ≥ 0, never significantly hurts, large where
  headroom). **AWR = task-dependent** (big wins + a real regression on precision-Square).
- **Effect smaller than LIBERO (+11) because these suites are shorter-horizon (fewer replans)**
  → consistent with the compounding law (gain ∝ #replans; see §5). Gain largest where base SR is
  low (headroom): stick-pull base 15.6 → +10.
- **Stat caveat:** wide CIs (few seeds); individual tasks ~1–1.5σ (suggestive). Claim rests on
  aggregate + direction-consistency. **[PENDING] add seeds on box-close/stick-pull/disassemble.**
- **[PENDING]** investigate Square AWR −4.8 (precision task → distribution-shift, or bug, or
  recoverability-boundary content). Lift not yet closed.

================================================================================
## 4. RESULTS — DEFENSIVE: learned value / RL does NOT beat simple selection
================================================================================
(Contained subsection: preempts "why not proper offline-RL / a learned verifier?". All confirm
per-state value is noise.)
- **chunk-Q value-BoN (Q-chunking QC): 0.476 ± 0.02** (below base 0.581). Diagnostic: critic is
  well-calibrated cross-state (val AUC 0.89, calib gap +0.534) but within-state Q std = **0.014**
  (~3% of cross-state spread), spearman(Q,vote) = −0.019 ≈ 0 ⇒ Q ≈ V(state), cannot rank
  candidates within a state. Causes: episode-success labels are a STATE property; vote-selected
  training data lacks outliers.  [DONE]
- **AWR + learned V(s) baseline: 0.652 ± 0.028** (≈ / slightly below const-baseline 0.684).
  Advantage-reweighting dilutes the uniform mode-distillation that drives the gain.  [DONE]
- **IQL (chunked offline, expectile-V + TD-Q + AWR extraction): 0.622 ± 0.023 (β=3),
  0.618 ± 0.011 (β=1)** — BELOW plain AWR (0.66–0.68). Even proper conservative offline-RL
  underperforms imitation. The chain **uniform imitation 0.68 > succ−V baseline 0.652 >
  Q−V (IQL) 0.62**: smarter value-weighting → WORSE (per-chunk advantage is noise).  [DONE]
  ([PENDING] IQL @ 100 ep for a matched-epoch bulletproof number; expected ≈0.65, still < AWR.)
- **naive on-policy signed policy-gradient: 0.182** (COLLAPSE) — needs strong trust-region;
  reported as "naive RL is unstable here"; conservative retry paused.  [DONE]
- oracle plan-selection headroom (isolate, held-out, luck removed) = **+0.024** per single
  chunk-decision (small); pass@8 = +0.18 is optimistic (continuation luck).  [DONE]

**Takeaway of §4:** the deployed BoN gain (+0.11) is NOT per-state value-ranking (that's ≈0);
it is **compounded outlier-rejection** — a small per-chunk +0.024 applied at every replan.

================================================================================
## 5. MATHEMATICAL FRAMEWORK — "A Model of Compounding"  (see model_of_compounding.tex)
================================================================================
Notation: episode horizon T, executed length R, **H = ⌈T/R⌉ replans**; chunk a=z_{1:K};
episode return R(s,a)∈{0,1}; value V(s); advantage A=R−V; per-step catastrophe (unrecoverable)
prob ρ.

1. **Oracle (no per-state signal):** Δ(s,c) = Q(s,c) − Q(s,c_def), E_s[Δ] ≈ 0. Equivalently the
   per-step failure rate ρ(s,c) ≈ ρ(s) — independent of the budget choice c (replanning corrects
   a single cheap decision).

2. **Selection = mode-seeking / tail-risk removal (NOT variance reduction by averaging):**
   consensus ĥa = argmax_i Σ_j κ(a_i,a_j) → argmax_a π(a|s) as N→∞ (mode). Per-step gain =
   removed tail risk: **g = u(ĥa) − E_{a~π}[u] = ρ(u_+ − u_−) = ρΔ**. Failure rate drops
   ρ → ρ_N ≈ P(Binom(N,ρ) ≥ ⌈N/2⌉) ≈ C(N,⌈N/2⌉) ρ^⌈N/2⌉ ≪ ρ.
   (IMPORTANT wording: it is mode-seeking + asymmetric tail removal, NOT symmetric variance
   reduction and NOT quality-ranking. A reviewer will push on "variance reduction".)

3. **Compounding law:** SR ≈ (1−ρ)^H (catastrophe-free); **ΔSR = (1−ρ_N)^H − (1−ρ)^H ≈
   H(ρ−ρ_N)**. Tiny per-step gain × H replans, because ONE catastrophe anywhere fails the episode.
   **Falsifiable prediction: ΔSR ∝ H = #replans × per-step head-room (ρ−ρ_N).**

4. **Asymmetry (why adaptivity can't compound):** budget choice leaves ρ unchanged (from 1) ⇒
   multiplies zero ⇒ ΔSR ≈ 0; selection reduces ρ ⇒ compounds. Same H factor, opposite result.

5. **AWR loss (distillation):** from max_π E[A]−β·KL(π‖π_D) ⇒ π*∝π_D·exp(A/β); training loss
   L_AWR = −E[w(s,a) Σ_k log π_θ(z_k|z_{<k},s)] + β_KL·KL(π_θ‖π_ref), with w=min(exp(A/β),w_max),
   A=R−V. Since w≥0, AWR only RAISES likelihood ⇒ imitation-bounded (explains the ~0.68 ceiling).

6. **Cost model (amortized vision):** C_ep(single)=H(C_vis+K·c_AR); C_ep(BoN-N)=H(C_vis+N·K·c_AR)
   → H·C_vis when C_vis≫c_AR. BoN adds cost only on the cheap AR axis; distillation removes even
   that. **[PENDING] batch=1 latency N-sweep to confirm ~flat wall-clock.**

CAVEAT to state in the paper: this is a first-order MODEL (safe/catastrophic dichotomy, small-ρ),
meant to explain the DIRECTION and the ΔSR ∝ #replans prediction — not exact numbers (the
+0.024 single vs +0.11 deployed is not a clean ×34; effects saturate).

================================================================================
## 6. THE CENTRAL RECONCILIATION (must be explicit in the paper)
================================================================================
Apparent contradiction: "per-state selection doesn't work" (adaptivity, chunk-Q) vs "best-of-N
works". Resolution = **optimization vs tail-removal**:
- Per-state **optimization** (choose the best budget/chunk) needs a per-state "which is best"
  signal → ≈ 0 (oracle) → fails (adaptive K/R, chunk-Q).
- best-of-N is **not** optimization — it **rejects the rare catastrophic outlier** (mode-seeking).
  Each rejection is worth little (+0.024) but **compounds** over ~H replans (+0.11) because one
  catastrophe fails the episode.
- **Same compounding** that KILLS adaptivity (single decisions washed) MAKES selection work
  (outlier-avoidance accumulates). Negative and positive = one mechanism, not a contradiction.

================================================================================
## 7. RELATED WORK / POSITIONING (for the Related Work section)
================================================================================
- **OAT** (RSS 2026) — the substrate; explicitly leaves adaptive autoregressive depth as open,
  "grounded in uncertainty/information, not ad-hoc heuristics." We answer it (negatively for
  reduction, positively for selection).
- **Test-time scaling / selection:** RoboMonkey [2506.17811] (sample+Gaussian+VLM verifier),
  MG-Select [2510.05681] (verifier-free BoN), RoVer [2510.10975]. Our wedge: ordered/
  prefix-decodable action tokens + amortized-vision (BoN nearly free) + the compounding diagnosis.
- **Ordered-token test-time search:** **SoTo** [2604.15453, ICML 2026] does BoN/beam/lookahead
  over ordered tokens for IMAGE generation. We are the closed-loop CONTROL counterpart: use it to
  motivate the **beam-search negative-transfer** result ([PENDING], see §9).
- **Adaptive horizon (R-axis):** PACE [2606.00537], DEHP [2606.11408], AAC [2604.04161] report
  gains on contact-rich tasks; we show their signals lose to a random control on LIBERO (proper
  matched-cost + random baseline they omit) → scope: forgiving vs unrecoverable regimes
  (recoverability boundary).
- **RL fine-tuning for VLA:** IQL/CO-RFT [2508.02219], Q-chunking [2507.07969], V-GPS/VGAS,
  TGRPO, RL-Token [2604.23073], BORA [2605.30226], credit-assignment survey [2604.09459]. We use
  IQL as the strong offline-RL baseline (underperforms imitation here).
- **NOVELTY = the diagnosis + compounding model.** Methods (BoN, AWR, IQL) are borrowed and used
  as demonstrations, stated honestly.

================================================================================
## 8. SCOPE / LIMITATIONS (state honestly)
================================================================================
- **Single policy architecture (OAT).** Multi-suite (3 benchmarks, 2 simulators) ≠
  multi-architecture. The MECHANISM (compounding, selection) is a property of the closed-loop
  chunk-policy CLASS; the DEMONSTRATION is OAT-only. Future work: Diffusion Policy, ACT, π0.
- **K-axis is OAT-specific** (needs an ordered/prefix-decodable tokenizer); **R-axis, compounding,
  selection, amortized-vision are generic** to chunking policies.
- **Simulation only** (LIBERO/robomimic/MetaWorld); no real robot.
- The adaptivity-negative is mostly LIBERO-centric (oracle/controllers on LIBERO); the SELECTION
  positive is confirmed multi-suite. Do not over-claim "both effects across all suites."
- The compounding model is first-order (directional prediction, not exact numbers).

================================================================================
## 9. PAPER OUTLINE (section plan for the writing agent)
================================================================================
1. **Introduction** — the compute-allocation question for action-chunking policies; the two-part
   answer (reduction fails / selection wins); contributions (diagnosis + model + demonstration).
2. **Setup** — OAT, the K and R axes, the counterfactual-oracle method, benchmarks.
3. **§ Reduction doesn't pay (NEGATIVE).**
   3.1 Oracle: per-state value(K), value(R) ≈ 0 (MONEY SHOT figure).
   3.2 Controllers ≈ random/fixed at matched cost (K: predictor=agnostic-mix; R: convergence,
       PACE < random). Include the sustained agnostic-mix + random controls.
   3.3 Mechanism: compounding washes single decisions (bridge to §5).
4. **§ Selection pays (POSITIVE).**
   4.1 Best-of-N (vote) exceeds base; scaling law; amortized-vision cost.
   4.2 Distillation into single-forward (AWR); imitation ceiling.
   4.3 Multi-suite (LIBERO/robomimic/MetaWorld); gain scales with #replans × headroom.
5. **§ A Model of Compounding** — the math (§5 here); the ΔSR ∝ #replans prediction + its
   empirical confirmation (multi-suite).
6. **§ Can learned value do better? (DEFENSIVE, short / partly appendix)** — chunk-Q, AWR+critic,
   IQL, signed-PG all ≈/< simple selection → per-state value is noise; gain is compounded
   outlier-rejection.
7. **Related work** (§7).  8. **Limitations** (§8).  9. **Conclusion** — spend compute on
   selection (compounds), not per-observation reduction (washed out).
(Ablations to appendix: pow2 err(k) curve; mode-decomposed BoN — coarse-to-fine, first-token-temp;
beam-search negative-transfer [PENDING]; full multi-suite per-task tables; RL/value details.)

================================================================================
## 10. FIGURES / PLOTS TO BUILD (for the figures agent)
================================================================================
- **F1 Teaser:** one schematic — reduction (fewer K/R) washed by replanning vs selection
  (best-of-N, amortized vision) that compounds. Two-panel: "washed out" vs "compounds".
- **F2 MONEY SHOT — oracle value:** bar/violin of per-state value(K) and value(R) ≈ 0 across
  tasks (from oracle_grid_mt); annotate value_k=+0.043, value_R≈0; contrast with the SR(k4→k8)
  and SR(R) gaps to show "compounding, not per-state".
- **F3 BoN scaling law:** SR vs N (1/4/8/16/32) with error bars; horizontal base line; annotate
  plateau ~0.72; log-N fit.
- **F4 Fixed-R sweep + adaptive-R controls:** SR vs R (8/16/24/32); overlay convergence, PACE,
  pace_raw, random at matched mean (points) — show all adaptive < random.
- **F5 K-axis Pareto:** SR vs mean tokens — fixed {1,2,4,8}, predictor w2/w4, entropy,
  agnostic-mix; show fixed k=4 dominates and predictor≈agnostic-mix.
- **F6 Multi-suite bars:** per-task base / BoN / AWR (Can, 4×MW, Square) with error bars;
  aggregate Δ. Highlight BoN never-negative; AWR Square regression.
- **F7 Compounding prediction:** ΔSR (BoN gain) vs #replans per suite/task; overlay the model
  line ΔSR ∝ H; (secondary axis or color = base-SR headroom).  [needs exact #replans — PENDING]
- **F8 AWR distillation:** SR vs epochs (N=8 and N=16 sources) with base + BoN-ceiling lines;
  show the ~0.68 imitation ceiling.
- **F9 (defensive) learned-value fails:** bar of base / vote-BoN / chunk-Q value-BoN / IQL /
  AWR+critic; + inset: within-state Q std (0.014) vs cross-state (0.53) to explain why.
- **T1 (table)** full multi-suite per-task numbers. **T2 (table)** RL/value negatives.

================================================================================
## 11. STATUS — running / pending experiments (will add to the paper)
================================================================================
- **[RUNNING/PENDING] RoboCasa** — retrain OAT (tokenizer+policy) + BoN/AWR; robosuite, cheapest
  port; strengthens breadth. Guide: `ROBOCASA_BON_AWR_PLAN.md`. Adds a 4th benchmark at
  camera-ready.
- **[PENDING] RoboTwin** — bimanual (14D action) + SAPIEN; harder port; model side is config +
  retrain (action_dim 7→14, maybe num_registers). Reserve for camera-ready / journal.
- **[PENDING] batch=1 latency N-sweep** (`measure_latency_adaptive.py`) → confirm BoN ~flat
  wall-clock → upgrade abstract to "negligible added latency" + a latency table.
- **[PENDING] beam-search negative-transfer** (`predict_action_beam`, likelihood/consensus scored)
  → show ≈/< flat BoN → the SoTo contrast (images: search helps; closed-loop control: it doesn't).
- **[PENDING] IQL @ 100 ep** (matched-epoch bulletproof number, expect ≈0.65 < AWR 0.68).
- **[PENDING] more seeds** on multi-suite suggestive tasks (box-close, stick-pull, disassemble)
  for significance; **investigate Square AWR −4.8**; **close Lift**.
- **[PENDING] exact mean_replans** per suite (add a runner log line) → for F7.
- **[RESERVED, rebuttal only] sustained-oracle-R** — the formal upper bound closing the
  "sustained per-step adaptive policy" objection (expensive; agnostic-mix already covers it).

================================================================================
## 12. KEY FILES (what produced what)
================================================================================
- `abstract_aaai27.txt` — final abstract (v4) + title + optional sentences.
- `model_of_compounding.tex` — the math (compilable standalone; drop into aaai27.tex).
- Policy/inference: `oat/oat/policy/oatpolicy.py` (predict_action, predict_action_bon_free
  [vote/medoid/value], predict_action_variable_r [convergence/random/fixed/pace/pace_raw],
  predict_action_predictor/agnostic).
- Oracle/counterfactual: `oat/scripts/branch_value_k.py` (grid, bon_n pass@k, bon_isolate).
- BoN/eval: `oat/scripts/eval_policy_sim.py` (--bon_free, --bon_signal, --chunk_q, --adaptive_r,
  --n_action_steps, --agnostic_mix, --token_predictor).
- AWR: `oat/scripts/collect_awr_dataset.py`, `oat/scripts/train_awr.py` (--critic for V(s) baseline).
- Value/RL: `oat/oat/model/chunk_q.py` (ChunkQ + ChunkV), `oat/scripts/train_chunk_q.py`,
  `oat/scripts/train_iql.py`, `oat/scripts/train_pg.py`, `oat/scripts/diag_chunk_q.py`.
- Diagnostics: `oat/scripts/diag_convergence_div.py`, `oat/scripts/gate_disagreement.py`,
  `oat/scripts/analyze_criticality.py`, `oat/scripts/per_timestep_recon_error.py`,
  `oat/scripts/calib_pace_threshold.py`, `oat/scripts/measure_latency_adaptive.py`.
- Ports: `oat/ROBOCASA_BON_AWR_PLAN.md`, `oat/ROBOMIMIC_BON_AWR_GUIDE.md`,
  `oat/scripts/convert_robocasa_dataset.py`, config templates under
  `oat/oat/config/task/{tokenizer,policy}/robocasa/`.
- Full research log with every number and dead-end: `CLAUDE.md` (§"Active research").

================================================================================
## 13. HEADLINE NUMBERS (quick reference)
================================================================================
base 0.581 | BoN N=8 0.690 (+0.11) | BoN N=16 0.712 (plateau ~0.72) | AWR-distill 0.684 (+0.10,
deployable) | fixed k=4 0.496, k=8 0.58 | fixed R=8 0.635, R=16 0.577, R=32 0.440 |
oracle value_k +0.043, value_R ≈0 | convergence-R 0.553 < random 0.595 | PACE 0.482 < random 0.537
| chunk-Q value-BoN 0.476 | IQL 0.62 < AWR 0.68 | multi-suite aggregate Δ_BoN +4.1pp, Δ_AWR +4.9pp.
