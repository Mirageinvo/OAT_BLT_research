---
name: ICRA matched paper plan
overview: "RM/MW matched-only eval for ICRA. All benches same protocol. No exploratory numbers in paper. Lift = retrain later; Square = wait for baseline; NOW = Can + MW specialists."
todos:
  - id: phase0-docs
    content: "Claims = matched-only; kill exploratory paper roles"
    status: completed
  - id: phase1-script
    content: cluster_matched_triplet.sh exists
    status: completed
  - id: phase2-now
    content: "NOW rematch: Can, coffee, stick, disassemble, box-close"
    status: pending
  - id: phase2-square
    content: "After square baseline done — matched BoN+AWR"
    status: pending
  - id: phase2-lift
    content: "After Lift retrain — full matched triplet on new ckpt"
    status: pending
  - id: phase4-tables
    content: "Fill matched Table B + Table C from matched artifacts only"
    status: pending
  - id: phase5-latency
    content: "Extend latency script for bon_free; Table C all paper suites"
    status: pending
  - id: phase6-libero-opt
    content: "OUT OF SCOPE (PI)"
    status: cancelled
  - id: phase7-breadth
    content: "OUT OF SCOPE (PI)"
    status: cancelled
isProject: false
---

# ICRA — matched eval build plan (RoboMimic / MetaWorld)

Canonical: [RESOLUTIONPLAN.md](RESOLUTIONPLAN.md).

## Paper rule

- **In paper:** only `output/eval/matched/<suite>/` (baseline / BoN / AWR / latency).
- **Not in paper:** chain5↔exploratory BoN/AWR, old SR, any “controlled negative” from pre-matched runs.
- **All suites same recipe** — Can, coffee, stick, disassemble, box, Square, Lift. No special coffee story.

## NOW vs LATER

| Wave | Suites |
|------|--------|
| **NOW** | Can, coffee-pull, stick-pull, disassemble, box-close |
| **LATER** | Square — when matched baseline finishes |
| **LATER** | Lift — after policy **retrain**, then full matched triplet |

## Protocol (all)

`test_start_seed=1000`, `n_test=50`, `-n 3`, OAT8, BoN `--bon_free 8 --bon_signal vote`.  
Δ = method − matched baseline.

```bash
SUITE=<name> GPU=<id> bash scripts/cluster_matched_triplet.sh
# box-close:
OUT_ROOT=output/eval/matched/metaworld_box-close SUITE=box-close GPU=<id> \
  bash scripts/cluster_matched_triplet.sh
```

### NOW detail

1. **Can / coffee / stick** — matched baseline DONE → run triplet (BoN+AWR eval). If paper policy is “no reuse of pre-matched AWR weights”, collect→train AWR first then only AWR matched eval; otherwise rematch eval on existing distill is inference-only.
2. **disassemble / box** — matched BoN now; then collect→train AWR → matched AWR (no prior paper AWR).
3. Write SR into Table B from `summary.json` / `eval_log.json` only.

### After NOW

- Square baseline → same BoN+AWR matched.
- Lift retrain → new ckpt → matched baseline+BoN+AWR (discard old Lift matched baseline for paper if ckpt changes).
- Phase 5 latency + Table C for **every** suite kept in paper.

## Phase 5 — Latency + Table C

**Full protocol (canonical):** [`RESOLUTIONPLAN.md`](RESOLUTIONPLAN.md) § Latency / Table C (2026-07-17).

Extend `my_scripts/measure_latency_*` for BoN N=8 + AWR. Policy-forward **batch=1**.  
Per suite: Single / BoN / AWR ms → `matched_s10000/<suite>/latency.json` → Table C (SR from Table P × ms × ΔSR × cost).

Must: same cluster GPU/docker as SR eval; deterministic as possible; **5–10** timed reps → median + mean±std; same obs preprocess as `eval_policy_sim`; warmup excluded; **AWR ckpt = Wave2 `awr_s10000_<suite>.ckpt`** (same arch); **`latency.json` records `git_commit` (+ dirty/branch/host/gpu/ckpt paths)**.  
Does **not** re-run matched SR.

## Success

- [ ] Matched BoN+AWR for Can, coffee, stick, disassemble, box  
- [ ] Square matched after baseline  
- [ ] Lift matched after retrain  
- [ ] Table B + Table C from matched only  
- [ ] No exploratory Δ in paper text/tables  
