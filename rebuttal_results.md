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

- **72A3 (does selection beat the single-sample policy under a clean protocol?)** Yes: +13.1 pp [10.2, 16.0], every task.
- **Equal-N control (is it just N=8 sampling?)** No: uniform choice among 8 candidates is indistinguishable from
  N=1 (−1.5 [−4.7, +1.5]); CS-8 beats Random-8 by +14.7 [11.9, 17.5]. The gain comes from the selection rule.
- **NTCF (ablations)** Scoring the whole executed prefix matters: endpoint-only scoring loses 6.9 pp [4.4, 9.3].
  The adaptive bandwidth is not required: a bandwidth frozen on held-out seeds is within −1.2 [−3.2, +0.8].
- **9QTC (CS vs medoid)** Medoid and CS choose differently in 11% of decisions (in 87% of episodes at least once),
  but their SR is statistically indistinguishable (−0.5 [−2.3, +1.2]). Honest reading: the benefit is from
  consensus (mode-seeking) selection over the prefix; this experiment does not show an advantage of the
  kernel score over medoid. Do not claim CS > medoid.

## Not to be used

- Timing columns: all variants ran concurrently (MAXJ=6) and medoid ran alone afterwards (hence its ~half ms);
  latency answer comes only from the separate idle-GPU measurement (P2).
- Old (pre-v3) LIBERO-LONG numbers as paired evidence; direction/size of the reset bug's effect on them is unknown.
- `h median 2.150` in the disagreement line is the median over decisions where CS and medoid differ, not the
  calibration value (H_FIXED = 0.9926); don't quote it as "the" bandwidth without checking.

## Draft rebuttal text (English)

During validation of our LIBERO-LONG evaluation, we found two reset issues. First, a pending worker auto-reset could
discard the first action chunk after an explicit reset. Second, in three tasks with articulated fixtures, the LIBERO
version we used accumulated fixture-initialization samplers across hard resets. The number of resets in a reused
environment depended on when earlier episodes ended, so identical reset seeds did not guarantee matched starting
states across methods. We corrected both issues and reran the comparisons with the same frozen checkpoint and
candidate-generation settings on 500 shared initial scenes with three generation seeds. We verified that initial
simulator-state hashes matched across all methods and repeats.

Under this protocol, CS-8 reaches 71.1% versus 58.0% for the single-sample policy (+13.1 pp, paired 95% CI
[10.2, 16.0]), improving on all ten tasks. Choosing uniformly at random among eight candidates gives 56.5%, no
different from a single sample (−1.5 pp, [−4.7, 1.5]), so the gain comes from the selection rule rather than from
drawing more samples (CS-8 vs Random-8: +14.7 pp, [11.9, 17.5]). Scoring only the chunk endpoint instead of the
executed prefix lowers success by 6.9 pp ([4.4, 9.3]), while freezing the bandwidth on held-out seeds changes it by
−1.2 pp ([−3.2, 0.8]). A prefix medoid selects a different candidate in 11% of decisions but reaches the same
success rate (−0.5 pp, [−2.3, 1.2]); we therefore attribute the improvement to consensus selection over the executed
prefix and will state that the kernel score and the medoid perform equivalently in this setting.

## Pending

- Table 2 (P1): run `scripts/rebuttal/export_kr.py` on `oracle_grid_mt.npz`, report improved/same/worse counts only.
- P2 latency on an idle GPU.
