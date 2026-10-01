# Selector baselines — status ledger (branch `aamas27_selector_baselines`)

Updated: **2026-10-01 10:20 MSK**. Scope: RoboMimic / MetaWorld / RoboCasa. LIBERO = supervisor only.

**1 Oct — CS vs KDPE (не смешивать [E]/[C]):** числа и CI в [`CS_VS_KDPE_TABLE.md`](CS_VS_KDPE_TABLE.md) и дамп [`AAMAS27_DUMP.md`](AAMAS27_DUMP.md) §2.5–2.6. aic4: 2 evals/GPU; GPU0 coffee+disassemble **[C]** vote; GPU1 stick vote **[C]** + lift kdpe **[C]**; GPU2 stick kdpe **[C]** + coffee kdpe **[E]**. Старые GPU1-lane в SIGSTOP.

**Progress (paper cells):** **40 / 44** done · **2** running on aic4 (close max_ll, mw vote) · **2** queued (mw medoid/max_ll) · KDPE 2/3 RoboMimic  
**Hosts:** ccm EGL 2/GPU `n_par=2` · aic4 GPU2 OSMesa ≤2 RC `n_par=1` (GPU0/1 = VLA fits only)

**⚠ ccm RoboCasa looks systematically low on some tasks** (coffee 34 vs aic4 50; faucet medoid 41.6 vs aic4 56.8; microwave random 25.2 vs aic4 43.6). ccm close-drawer matches the paper. Until this is explained, **aic4 is the reference host for RC**.

---

## 0. vote vs PDF CS-8

| Task | PDF CS-8 | ours vote | Δ | Verdict |
|------|----------|-----------|---|---------|
| can | 85.6±2.2 | **85.6±3.6** | 0.0 | **pass** |
| square | 31.0±2.8 | 32.0±3.2 | +1.0 | pass |
| coffee-pull | 43.2±4.8 | 44.4±3.8 | +1.2 | pass |
| box-close | 67.2±4.1 | 69.6±1.7 | +2.4 | ok |
| stick-pull | 25.6±3.6 | **26.8±4.1** | +1.2 | **pass** (recheck) |
| lift | 92.7±0.6 | **95.2±2.7** | +2.5 | **ok** |
| disassemble | 62.4±4.6 | 65.6±4.3 | +3.2 | ok |
| close-drawer (RC) | 56.4±2.3 | **56.8±5.6** | +0.4 | **pass** |
| coffee-button (RC) | 55.6±2.7 | aic4 **50.0±4.2** (ccm 34.4 discarded) | −5.6 | ok (−2σ; ccm artifact confirmed) |
| faucet-off (RC) | 56.0±3.4 | **52.4±9.1** | −3.6 | ok (noisy) |
| microwave-off (RC) | 54.8±2.5 | — | — | vote killed (hung); restart pending |

Protocol: N=8 K=8 T=1.0 topk=10, `-n 5`×50, `test_start_seed=10000`.

---

## 1. Checklist

| Category | Status | Notes |
|----------|--------|-------|
| Selectors code | done | |
| SR RoboMimic | **done** | lift = ep-1400 (`lift1400/`) |
| SR MetaWorld | **done** | stick recheck pass |
| SR RoboCasa | **partial** | **12/16** cells trusted; close max_ll + mw vote live on aic4; mw medoid/max_ll queued |
| Latency | not started | unloaded H100 later |

---

## 3. Master table — SR

### 3.1 RoboMimic (aic4)

PDF columns = Table 4 of `oat-aaai.pdf` (± = **SE** across 5 seeds). Our columns: ± = **std** across 5 exps (SE = std/√5), all 5 exps on the same 50 scenes (seed 10000).


| Task | ckpt | PDF base | PDF CS-8 | our base | vote | random | medoid | max_ll | KDPE | Δvote | check |
|------|------|----------|----------|----------|------|--------|--------|--------|------|-------|-------|
| can | ep-1700 | 76.3±2.4 | **85.6±2.2** | **84.0±4.0** / paired 87.6±3.0 | **85.6±3.6** | 85.2±3.0 | 88.8±2.7 | 90.4±4.3 | 89.2±5.8 | 0.0 | **pass** |
| lift | ep-1400 | 90.5±1.3 | 93.2±1.2 | **96.0±3.7** | **95.2±2.7** | 96.4±1.7 | 96.8±3.6 | **96.4±3.0** | 94.8±3.3 | +2.0 | **ok** |
| square | ep-0700 | 30.7±1.0 | 30.5±2.2 | — | **32.0±3.2** | 30.0±7.9 | 29.6±4.3 | 34.4±7.4 | 32.4±7.8 | +1.5 | **pass** |

### 3.2 MetaWorld (aic4)

| Task | ckpt | PDF base | PDF CS-8 | vote | random | medoid | max_ll | KDPE | Δvote | check |
|------|------|----------|----------|------|--------|--------|--------|------|-------|-------|
| box-close | ep-2000 | 59.6±3.5 | 66.4±3.0 | **69.6±1.7** | 61.6±5.4 | 67.2±3.0 | 66.4±3.3 | 62.4±3.8 | +3.2 | **ok** |
| coffee-pull | ep-1000 | 40.8±2.3 | 43.2±4.8 | **42.8±3.3** (парно; старый 44.4±3.8) | 43.2±3.9 | 44.0±0.0 | 46.0±3.2 | 41.6±3.0 (jsonl [E] rerun GPU2) | −0.4 vs PDF | **ok** |
| stick-pull | ep-0800 | 15.6±6.2 | 25.6±2.6 | **26.8±4.1** | 16.4±2.2 | 18.8±2.7 | 14.8±7.4 | 18.0±4.2 | +1.2 | **pass** |
| disassemble | ep-1400 | 62.4±5.2 | 63.2±6.3 | **60.4±2.6** (парно; старый 65.6±4.3) | 56.8±4.6 | 63.2±3.0 | 64.0±4.9 | [E] kdpe queued GPU2 | −2.8 vs PDF | **ok** |

### 3.3 RoboCasa

| Task | ckpt | PDF base | PDF CS-8 | vote | random | medoid | max_ll | check |
|------|------|----------|----------|------|--------|--------|--------|-------|
| close-drawer | @0.700 | 56.0±1.1 | 59.6±1.6 | **56.8±5.6** ccm | **54.8±6.4** ccm | **53.6±3.6** ccm | **58.8±3.3** aic4 | vote **pass** |
| coffee-button | @0.600 | 44.0±3.0 | 55.6±2.7 | **50.0±4.2** aic4 | **52.4±5.4** aic4 | **51.2±4.1** aic4 | **50.0±4.2** aic4 | ccm ×4 **discarded** (§6) |
| microwave-off | @0.620 | 43.6±3.1 | 51.6±3.0 | **46.4±4.6** aic4 | **43.6±9.2** aic4 (ccm 25.2 ✗) | **45.2±10.3** aic4 (ccm 28.8 ✗) | **39.6±7.7** aic4 (ccm ✗) | ccm **discarded** |
| faucet-off | @0.580 | 52.4±2.5 | 56.0±3.4 | **52.4±9.1** aic4 | **52.8±8.2** aic4 | **56.8±5.9** aic4 (ccm 41.6±2.6 ✗) | **49.6±9.9** aic4 | ok |

Per-exp coffee vote (aic4): .56 / .46 / .52 / .46 / .50.

### 3.4 Check summary

| Status | Count / tasks |
|--------|----------------|
| pass/ok | RM+MW all 7; close-drawer vote; coffee vote (aic4) |
| RC done (trusted) | close vote/random/medoid; coffee vote/random/medoid; faucet ×4; microwave random (**11/16**) |
| live | aic4: coffee max_ll, close max_ll · ccm: mw medoid/max_ll (not for paper) |
| queued aic4 | mw vote → mw medoid → mw max_ll |
| discarded | coffee ×4 ccm; faucet medoid ccm (41.6 vs 56.8); microwave ccm (random 25.2 vs 43.6) |

---

## 5. Live NOW (2026-09-29 23:25 MSK) — see «Очереди на aic4» below; older text in this section is stale

| Host | Jobs |
|------|------|
| aic4 GPU0 | VLA fit + base can (P1) → then base lift (queued) |
| aic4 GPU1 | VLA fit + KDPE square (Exp2+) + KDPE MetaWorld queue (box-close live; coffee+ get episodes.jsonl) |
| aic4 GPU2 | close max_ll (Exp5) · mw vote (Exp1); then mw medoid/max_ll via `/tmp/queue_rc_aic4_mw.sh` |
| ccm | mw medoid · mw max_ll (kept for ccm-vs-aic4 diagnosis only) |

Overnight incident (aic4): the old `fill_rc_aic4.sh` woke up and started faucet max_ll + mw random alongside the coffee queue → coffee random died at 1/50 (3rd concurrent OSMesa). Both old schedulers replaced by one queue (`queue_rc_aic4.sh`, log `logs/queue_rc_aic4.log`).

---

## 6. Coffee-button — dig + aic4 control (DONE)

**ccm (−n 5 @10000, n_par=2, EGL):** all selectors ~30–34% vs PDF 55.6 — **discard for paper**.

**Ruled out as root cause of the gap:** wrong ckpt/tok (sha256 = HF), selector-only bug (all 4 low), host-wide RC break (close-drawer vote 56.8 ≈ PDF on same ccm).

**aic4 control (2026-09-28, OSMesa, n_par=1, seed10000, −n 1):**

| mode | ours | paper seed10000 |
|------|------|-----------------|
| baseline | **40%** | 52% |
| vote | **64%** | 54% |

**aic4 full vote (−n 5 @10000):** **50.0±4.2** vs PDF 55.6 → ccm −21pp was a ccm-run artifact, not the locked policy.

**Same pattern on faucet medoid:** ccm 41.6±2.6 vs aic4 56.8±5.9 (same protocol). Differences ccm vs aic4: EGL vs OSMesa, `n_par=2` vs 1. Close-drawer on ccm is fine, so the defect is task-dependent. Root cause still open.

## 6b. KDPE-OAT (branch `aamas27/kdpe-oat` @ 4e81b91)

Deployed on aic4 as a separate code copy `~/oat_code_kdpe` (KDPE patch applied on top of aic4 `oat_code`, which carries the uncommitted random-selector / `n_test_vis` fixes; `PYTHONPATH=~/oat_code_kdpe`). Unit tests 17/17 pass on aic4. KDPE originally needed D=7 → RoboMimic only; MetaWorld D=4 added in 511fa6a (below); RoboCasa D=12 unsupported.

| step | can ep-1700 | notes |
|------|-------------|-------|
| offline diagnostic (512 obs, R=16) | **PASS** | no NaN/collapse; ties 1.8%; idx hist flat; KDPE≠vote 78%, ≠medoid 77%; pos term dominates (norm quad pos 30 vs rot 0.05 / grip 0.11) |
| smoke 1 seed × 10 eps (GPU1) | kdpe 7/10 · vote 9/10 | plumbing OK, params in `eval_log.json` |
| full −n 5 × 50 @10000, n_par=4 (GPU1) | **89.2±5.8** (.92/.82/.94/.94/.84) | vote 85.6±3.6 · max_ll 90.4±4.3 |

lift ep-1400 (GPU0): diagnostic **PASS** (ties 0.8%, KDPE≠vote 75%) → full **94.8±3.3** (1.0/.92/.96/.92/.94) vs vote 95.2±2.7.

square ep-0700 (GPU1): first try failed at diagnostic (square zarr missing on aic4) → pulled `square_N200.zarr` from HF `aaai-datasets`, symlinked into `data/robomimic/` → diagnostic **PASS** (ties 0.2%, KDPE≠vote 78%) → full *live* (started 17:28 UTC).

Ckpts verified = paper: sha256 of aic4 can ep-1700 / lift ep-1400 / square ep-0700 match HF `aaai27-models/checkpoints/selected_from_output/...` (same files as the vote/random/medoid/max_ll cells).

**Paired bootstrap (per-exp SR, 5 vs 5, 100k resamples + exact permutation).** All 5 exps share the same 50 init states (`test_start_seed+i`), so exps differ only in sampling; per-episode outcomes are not logged, so pairing is per-exp only. Per-exp sd ≈3.4pp → SE(Δ)≈2.1pp, MDE≈6pp.

| task | Δ vs vote | 95% CI | perm p |
|------|-----------|--------|--------|
| can kdpe | +3.6 | [−2.0, +8.8] | 0.32 |
| can max_ll | +4.8 | [+0.4, +9.2] | 0.12 |
| can medoid / random | +3.2 / −0.4 | — | 0.21 / 1.00 |
| lift kdpe | −0.4 | [−3.6, +3.2] | — |
| can+lift kdpe (macro) | +1.6 | [−1.6, +4.8] | — |

→ on RoboMimic no selector is separable from vote at this n.

**P1 — base (no BoN) on can in the same pipeline** (`/tmp/run_base_cell.sh can`, GPU0, started 17:51 UTC, ETA ~19:35 UTC). Decides random 85.2 vs PDF base 76.3: base ≈85 → PDF base not comparable to the current pipeline; base ≈74 → candidates in the BoN path are not i.i.d. / BoN-path bug.

**MetaWorld (4-D) — commit 511fa6a (local, not pushed).** `kdpe.py` accepts D∈{4,7}; for D=4 (`[dx,dy,dz,grip]`) the rotation term is 0. Tests 22/22 on aic4. Diagnostic tie metric split: ties between *identical* endpoint actions (duplicate token sequences, harmless) vs ties between *distinct* endpoints (index-order dependent) — stop condition now uses the latter. disassemble zarr on aic4 was incomplete (no `meta`) and stick-pull was missing → both pulled from HF `aaai-datasets` into `~/hf_aaai_datasets`, symlinked.

| task | ties all / distinct | collapsed | off-diag kernel | KDPE graded pairs | diagnostic |
|------|---------------------|-----------|-----------------|-------------------|------------|
| box-close | 15.4% / **0.0%** | 0 | 0.48 | 79% (dup 2%) | **PASS** |
| coffee-pull | 23.4% / **0.4%** | 0 | 0.42 | — | **PASS** |
| stick-pull | 12.3% / **2.0%** | 0 | 0.28 | 55% (dup 1%) | **PASS** |
| disassemble | 21.3% / **0.0%** | 0 | 0.81 | 65% (dup 4%) | **PASS** |

Kernel is graded, not binary (most pairs have 1e-3<k<0.999; exact duplicates only 1–4% of pairs; KDPE picks the largest duplicate group in only ~50% of obs that have one) → density, not duplicate-counting. Smoke box-close (1×8 eps): 5/8, OK. Full queue `/tmp/queue_kdpe_mw.sh` on GPU1, sequential box-close → coffee-pull → stick-pull → disassemble (started 18:05 UTC; log `kdpe_n8/queue_mw.log`).

## 7. Next

1. aic4: coffee max_ll + close max_ll live (~5h).
2. Then aic4 queue 2 (`/tmp/queue_rc_aic4_mw.sh`): microwave vote/medoid/max_ll (ccm microwave numbers not for paper; mw random already live on aic4).
3. Latency when unloaded.

## Очереди на aic4 (переписаны 29 Sep, ~21:55 MSK)

Старые `/tmp/queue_*.sh` убиты (логи внутри `-o` удалялись `--force`, ретраев не было, ожидание по строке `DONE` с таймаутом 180 мин). Новые скрипты лежат в `~/queues/` на aic4, полосы запущены через `setsid nohup`. Проверка состояния: `bash ~/queues/status.sh`, журнал: `~/oat_eval_out/queues.log`.

Общая логика: маркер завершения = `eval_log.json`. Если наблюдаемый процесс пропал, а маркера нет, полоса перезапускает ячейку (до 2 раз). Свои ячейки ретраятся до 3 раз. Логи paired-ячеек лежат рядом с output-папкой: `~/oat_eval_out/paired/<task>/<method>.log`.

| Полоса | GPU | Порядок |
|---|---|---|
| g0 | 0 | can base (ждёт) → lift base → can base_ep ‖ can vote_ep → can kdpe_ep → box-close vote_ep → box-close kdpe_ep |
| mw | 1 | box-close KDPE (ждёт) → stick-pull KDPE → stick-pull vote_ep → coffee-pull KDPE → disassemble KDPE |
| sq | 1 | square KDPE (ждёт) → square vote_ep → square kdpe_ep |
| rc1 | 2 | microwave vote (ждёт) → microwave max_likelihood |
| rc2 | 2 | microwave medoid (ждёт) |

Для paired нужны jsonl с обеих сторон: у уже идущих square KDPE и box-close KDPE per-episode логов нет. Поэтому в очередь добавлены `square vote_ep/kdpe_ep` и `stick-pull vote_ep`. Если время поджимает, `square kdpe_ep` можно снять первым.

## Итоги 30 Sep, 00:50 MSK

Готово на aic4: can base **84.0±4.0** (SE 1.8; статья 76.3±2.4, наш vote 85.6) → на can селекция не даёт выигрыша в нашем пайплайне. Square KDPE **32.4±7.8** против vote 32.0 → паритет. Box-close KDPE **62.4±3.8** против vote 69.6 → Δ −7.2, нужна парная проверка (перезапуски с jsonl в очереди g0).

Очередь g0 вернули к исходному плану (`lane_g0c.sh`): lift base → can base ‖ vote → can kdpe → box-close vote → kdpe. Переезд парных ячеек на ccm отменён.

ccm (V100, EGL, код = копия `oat_code_kdpe` с aic4, `/workspace/oat/_kdpe_code`): проверка стека, can base/vote → lift base/vote, с jsonl. Выход: `/workspace/oat/output/eval/ccm_v100_paired/`. Первые повторы: can base 92, can vote 98. С aic4-числами напрямую не смешивать.

## Утро 30 Sep (10:40 MSK) — ночь прошла без падений

Парные раны (episodes.jsonl, одинаковые 50 сцен × 5 повторов), разница по сценам, bootstrap + sign-flip permutation:

| host | task | сравнение | SR | Δ | 95% CI | p |
|---|---|---|---|---|---|---|
| aic4 | can | vote − base | 88.8 vs 87.6 | +1.2 | [−4.4, +6.8] | 0.78 |
| aic4 | can | kdpe − base | 88.4 vs 87.6 | +0.8 | [−4.8, +6.4] | 0.89 |
| aic4 | can | kdpe − vote | 88.4 vs 88.8 | −0.4 | [−5.6, +5.2] | 1.00 |
| aic4 | square | kdpe − vote | 30.4 vs 38.8 | −8.4 | [−16.8, −0.0] | 0.08 |
| ccm V100 | can | vote − base | 92.0 vs 87.2 | +4.8 | [−0.4, +10.0] | 0.11 |
| ccm V100 | lift | vote − base | 96.8 vs 96.0 | +0.8 | [−2.4, +4.4] | 0.81 |

- **V100 не воспроизводит base 76.3 на can**: base 87.2 (aic4: 84.0 и 87.6). Число из статьи несравнимо с текущим пайплайном на обоих стеках.
- lift base 96.0 (aic4) / 96.0 (V100) против 90.5 в статье — тот же сдвиг.
- square vote в двух одинаковых ранах: 32.0 и 38.8 → межрановый разброс ~7 п.п.
- Остались: stick-pull vote (4/5), box-close vote (4/5) → box-close kdpe, затем KDPE coffee-pull и disassemble.

## Главные результаты ночи (зафиксировано 30 Sep, ~10:50 MSK)

1. **PDF base не воспроизводится** ни на aic4, ни на V100: can 76.3 → 87.6 (aic4) / 87.2 (V100); lift 90.5 → 96.0 / 96.0. Base из PDF — legacy другого пайплайна. RoboMimic-строки Table 4 пересчитываем целиком на нашем пайплайне (base, CS и KDPE на одинаковых 50 сценах).
2. **CS vs KDPE:** can — ничья (−0.4); square — KDPE −8.4 (CI [−16.8, −0.0], p=0.08); stick-pull — KDPE −8.8; box-close — предварительно −7.2. KDPE нигде не лучше CS и хуже на задачах, где есть запас по SR.
3. **Парное сравнение по сценам почти не сузило CI.** Доминирует разброс внутри сцены (сэмплинг политики), а не разброс между сценами → мощность растёт от числа повторов, а не сцен.
4. **Большой разброс между ранами** (square vote 32.0 vs 38.8). Причина — недетерминизм: для сэмплинга нет `torch.manual_seed`. Числа по 5 повторам ненадёжны.

### Приоритеты
- **P1 (идёт на aic4, старт ~10:42 MSK):** square base (GPU2), square vote2 (GPU0) и kdpe2 (GPU1) +5 повторов → слить в 10 exps и пересчитать парный CI; box-close vote → kdpe (GPU0); stick-pull vote → KDPE coffee-pull → KDPE disassemble (GPU1); после square base на GPU2 — lift vote → lift kdpe.
- **P2 (воспроизводимость):** в `eval_policy_sim.py` добавлен `--policy_seed` (exp i сидится `policy_seed + i`: random/numpy/torch/cuda + cudnn.deterministic), пишется в `eval_log.json` и в каждую строку `episodes.jsonl`. По умолчанию выключен → очереди работают как раньше. Smoke: can vote, 2 раза с `--policy_seed 0` (2 exp × 8 эпизодов, изолированная копия `~/oat_code_seed`). Если совпадёт бит в бит → перепрогнать ключевые ячейки (square vote/kdpe, can vote/kdpe) с сидом. Если нет → в статье пишем «5 повторов на 50 фиксированных сценах».
- **P3: DONE (H100 GPU1, warmup=50, 8×20).** Selector-only CS 0.264/0.321/0.429 мс vs KDPE 0.535/0.676/0.680 мс при N=8/16/32; IQR CS и KDPE не пересекаются. Матрицы KDPE N×N (в т.ч. 32×32). Profiler: KDPE ~72 CUDA launches при любом N → launch-bound, не баг. Формулировка: §2.7 `AAMAS27_DUMP.md`.

## Вечер 30 Sep (19:10 MSK)

- **square (10 повторов, [E]):** CS 36.0 против KDPE 33.0, Δ +3.0, CI [−3.6, +9.8], p=0.42 — **не значимо**. Ночное +8.4 для CS по 5 повторам не подтвердилось (второй прогон: 33.2 против 35.6). Square base 32.4±3.3.
- **box-close:** CS 68.8 против KDPE 58.8, Δ +10.0, CI [+3.6, +16.8], p=0.005. **stick-pull:** 28.8 против 18.0, Δ +10.8, CI [+3.6, +18.0], p=0.005.
- **coffee-pull KDPE:** 41.6±3.0 (CS 44.4±3.8 непарно, Δ +2.8). **lift CS парно:** 94.8±1.1 против base 96.0 (Δ −1.2, CI [−5.2, +3.2]).
- **Сбой GPU2 около 14:30:** lift KDPE упал 3 раза (`CUDA error: invalid argument`), square CS с сидом завис. GPU2 проверен, в порядке; обе ячейки перезапущены 18:55.
- **Confirmatory:** square KDPE с сидом 33.6±7.9 ≈ [E] 33.0 (парно Δ +0.6).
- **GPU0 освобождён** (снят seed stick-pull CS/KDPE).
- **Latency H100 GPU1 (warmup=50, 8×20):** CS 0.264/0.321/0.429 vs KDPE 0.535/0.676/0.680 мс (N=8/16/32). KDPE 32×32 подтверждён; launches ~72 при любом N → launch-bound. IQR CS vs KDPE не пересекаются.
