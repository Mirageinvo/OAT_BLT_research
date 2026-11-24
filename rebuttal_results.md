# AAMAS-27 #2379 — rebuttal results (LIBERO-LONG, scene protocol v3)

Source: `eval_out/rebuttal_cs_ablation_v3/main/{vs_base,vs_cs_prefix,vs_random}_strict.md` (server),
code `eafb8e0` (eval) + `a9d9f76` (aggregator). Same frozen checkpoint, N=8, K=8, R=16, T=1.0, top-k 10,
500 shared initial scenes (seed 10000+, 10 tasks x 50), generation seeds 0,1,2 -> 1500 paired episodes per variant.
CI = 95% paired bootstrap, scenes resampled within task, equal task weight.

## Integrity (all strict checks passed, all three aggregations exit 0)

- 9000 episodes, 500 per variant x seed, 0 non-ok, 0 unfinished, 0 duplicates.
- Initial-state hash differs across variants: 0/1500 keys; across repeat seeds: 0/500 scenes; init_id->task ambiguity: 0.
- K=8, R=16 in every episode of every variant.
- cs_prefix executed index vs its shadow decision: 0/34669 mismatches.
- `medoid_prefix` was OOM-killed once (batch 51, seed 0, cgroup oom_kill=1) and rerun from scratch alone on the same
  commit; its scene hashes match the other variants (covered by the 0/1500 check).

## Success rate (%, 1500 episodes each; std over the 3 generation seeds)

| variant | SR | std | Δ vs base [CI] | Δ vs CS-8 [CI] | Δ vs Random-8 [CI] |
|---|---|---|---|---|---|
| base (N=1) | 58.0 | 1.4 | — | −13.1 [−16.0, −10.2] | +1.5 [−1.5, +4.7] |
| **CS-8 (cs_prefix, ours)** | **71.1** | 1.1 | **+13.1 [+10.2, +16.0]** | — | **+14.7 [+11.9, +17.5]** |
| Random-8 | 56.5 | 1.0 | −1.5 [−4.7, +1.5] | −14.7 [−17.5, −11.9] | — |
| CS endpoint (step R−1 only) | 64.3 | 3.5 | +6.3 [+3.4, +9.1] | −6.9 [−9.3, −4.4] | +7.8 [+4.9, +10.8] |
| medoid prefix | 70.6 | 2.2 | +12.6 [+9.5, +15.7] | −0.5 [−2.3, +1.2] | +14.1 [+11.3, +17.0] |
| CS prefix, fixed h | 69.9 | 0.6 | +11.9 [+8.8, +14.9] | −1.2 [−3.2, +0.8] | +13.5 [+10.7, +16.3] |

- CS-8 > base on all 10 tasks (per-task gains +4.7 to +19.3 pp). Discordant scene-episodes CS-only vs base-only success: 366 vs 169.
- Random-8 vs base: 264 vs 287 discordant; per-task mixed.
- Selector disagreement on the same 8 candidates (34669 decisions on CS-8 trajectories): CS vs medoid pick a
  different candidate in 11.3%; CS vs endpoint 60.6%; adaptive vs fixed h 21.7%. 1309/1500 episodes contain
  ≥1 CS≠medoid decision; median normalized distance between the two executed prefixes there = 0.62.

## Interpretation per reviewer item

| reviewer | what to show |
|---|---|
| **72A3** | CS-8 vs base +13.13 pp [10.20, 16.00]; higher point estimate on all ten tasks. Random-8 separately isolates the contribution of ranking. |
| **NTCF** | Scoring the whole executed prefix vs the endpoint only: +6.87 pp [4.40, 9.34] — the clearest ablation. Adaptive vs held-out-calibrated fixed bandwidth: no clear advantage detected (+1.20 [−0.80, 3.20] for adaptive). |
| **9QTC** | CS and medoid select different executed prefixes in 11.3% of replans on CS trajectories (≥1 such replan in 87.3% of episodes), yet the SR difference is +0.53 pp [−1.20, 2.33] in favour of CS. They sometimes choose differently, but an SR advantage of the Gaussian kernel over the medoid is not established. Do not claim CS > medoid. |

Wording rule: a CI containing 0 is "no convincing difference detected", never "no difference" / "equivalent" / "does not depend on".
Main pair with two decimals: CS-8 71.13% vs Random-8 56.47% = +14.67 pp [11.87, 17.47]; vs base 58.00% = +13.13 pp [10.20, 16.00].
No new runs before the rebuttal; remaining work = Table 2 K/R analysis.

## Not to be used

- Timing columns: all variants ran concurrently (MAXJ=6) and medoid ran alone afterwards (hence its ~half ms);
  latency answer comes only from the separate idle-GPU measurement (P2).
- Old (pre-v3) LIBERO-LONG numbers as paired evidence; direction/size of the reset bug's effect on them is unknown.
- `h median 2.150` in the disagreement line is the median over decisions where CS and medoid differ, not the
  calibration value (H_FIXED = 0.9926); don't quote it as "the" bandwidth without checking.

## Draft rebuttal text (English)

We also fixed two reset issues in our LIBERO-LONG evaluation that could leave methods with different starting states,
and reran all comparisons under a paired protocol: the same frozen checkpoint and candidate-generation settings, 500
shared initial scenes and three generation seeds, with initial simulator-state hashes verified to match across all
methods and repeats.

Under this protocol, CS-8 reaches 71.13% versus 58.00% for the single-sample policy (+13.13 pp, paired 95% CI
[10.20, 16.00]), with a higher point estimate on all ten tasks. Choosing uniformly at random among eight candidates
reaches 56.47%, and we found no convincing difference from a single sample (−1.53 pp, [−4.67, 1.53]); CS-8 exceeds
Random-8 by +14.67 pp ([11.87, 17.47]), so the gain is attributable to the selection rule rather than to drawing more
samples. Scoring the whole executed prefix rather than only its endpoint improves success by 6.87 pp ([4.40, 9.34]),
while we found no convincing advantage of the adaptive bandwidth over a bandwidth calibrated on held-out seeds
(+1.20 pp, [−0.80, 3.20]). CS and a prefix medoid select different executed prefixes in 11.3% of replans (at least
once in 87.3% of episodes), but the success-rate difference is +0.53 pp ([−1.20, 2.33]), so an advantage of the
Gaussian kernel over the medoid is not established; we will state this explicitly.

<details><summary>Longer reset description (only if a reviewer asks)</summary>

During validation of our LIBERO-LONG evaluation, we found two reset issues. First, a pending worker auto-reset could
discard the first action chunk after an explicit reset. Second, in three tasks with articulated fixtures, the LIBERO
version we used accumulated fixture-initialization samplers across hard resets. The number of resets in a reused
environment depended on when earlier episodes ended, so identical reset seeds did not guarantee matched starting
states across methods.
</details>

## Table 2 (P1): per-state distribution behind the averages

Source: `oracle_grid_mt.npz` (120 states, 10 tasks; M=5 continuations per branch; k 1 vs 8, R 8 vs 32).
Matches the published summaries: mean ΔK +0.043, mean ΔR −0.028, corr(ΔK, recon_gap) 0.08.
"Unchanged" = same success count out of 5 in both branches.

| change in one state | improved | unchanged | worse | mean |
|---|---:|---:|---:|---:|
| K 1→8 at long R | 24 (20.0%) | 81 (67.5%) | 15 (12.5%) | +4.3 pp |
| shorter R at K=8 | 13 (10.8%) | 89 (74.2%) | 18 (15.0%) | −2.8 pp |

With 5 continuations one outcome moves an estimate by 20 pp, so these counts cannot separate real heterogeneity
from noise. Use counts only (no sign-test p: coarse M, and states share episodes).

Draft text:

We reanalyzed the 120 states underlying Table 2. Increasing token depth improved the observed success count in 24
states, reduced it in 15, and left it unchanged in 81. Shortening the execution horizon improved it in 13 states,
reduced it in 18, and left it unchanged in 89. These are empirical differences from five continuations per branch.
They show why the average alone is insufficient, but do not establish how much of the variation a controller could
exploit. Our conclusion remains limited to the tested controllers, which did not show a reliable advantage over
matched random allocation.

## Pending

- (optional) `export_kr.py` split of the ties into 0/0, 5/5 and mid, if a reviewer asks what "unchanged" consists of.
- P2 latency on an idle GPU.
