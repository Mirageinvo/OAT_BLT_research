# Совет по структуре (AAMAS) — Composer

**Venue:** AAMAS (не AAAI). **Роль:** структура, приоритеты секций, что вынести в приложение, как честно подать протокол и числа.  
**Модель:** composer-2.5-fast.

---

## 1. Что AAMAS ждёт от этой работы

Для AAMAS сильнее, чем для «чисто ML»-venue, звучит **системный вывод о замкнутом цикле и распределении inference-бюджета**: когда адаптивное управление глубиной/горизонтом не даёт выигрыша, а test-time **selection** — да, и почему (компаундинг vs локальная «ценность» чанка). Механизм (KDE-мода, ablation селекторов) — поддержка тезиса, не отдельный «SOTA по одной задаче». Значит **лидировать диагностикой и дизайн-принципом**, а не таблицей «мы победили KDPE на шести MetaWorld-задачах».

---

## 2. Рекомендуемый каркас (8 страниц + приложение)

### 2.1. Введение (≈1 стр.)

- **Проблема:** prefix-decodable action chunks обещают anytime fidelity, но на практике бюджет inference часто тратят на K/R, а не на выбор среди кандидатов.
- **Тезис в одном предложении:** дополнительный compute лучше направлять на **consensus selection** среди N чанков при фиксированном энкодере, чем на эвристики адаптивной глубины/горизонта.
- **Вклад (3 пункта, без раздувания):**
  1. Диагностика K/R на LIBERO (Tables 1–3 в духе PDF) — адаптивность не бьёт случай/фикс при matched cost.
  2. Положительный результат: CS (vote) и масштабирование по N; **новая** ablation-линейка селекторов (LIBERO + RM/MW/RC где уместно).
  3. Разбор механизма: mode-seeking по траектории чанка vs endpoint-KDE vs likelihood vs random — с отсылкой к латентности (selector-only vs e2e).

Не начинать с RoboMimic base 76.3 vs 87.6 — это сразу подрывает доверие; это зона **Limitations**.

### 2.2. Preliminaries + setup (≈0.75 стр.)

- OAT, один forward энкодера, N сэмплов, R, K=8, N=8 по умолчанию.
- **Один блок «Evaluation protocol»** с явной таблицей по сьютам (исправить PDF): не «five training seeds», а **5 evaluation repeats × 50 fixed inits** (seed 10000) для RM/MW/наших прогонов; RoboCasa — 5 scene blocks; LIBERO-LONG — 500 rollouts, 5 exps; per-task checkpoints где применимо.
- Сразу определить метки **[E]** и **[C]** и правило смешивания (см. §4).

### 2.3. **Section 3 — Diagnosis: adaptive compute on K and R (lead with LIBERO)**

**Это должно идти до «наш CS побеждает».** AAMAS-читатель должен понять *почему* вы не продаёте ещё один adaptive controller.

- Сжато: Tables 1–3 (K-axis, R-axis, convergence/PACE-class на LIBERO) — из PDF, без перегруза цифрами в тексте.
- Якорная фраза: локальная ценность смены k или R **смывается replanning**; выигрыш от fidelity/horizon — **компаундинг по эпизоду**, не per-state сигнал.
- Связка с selection: тот же compounding объясняет, почему BoN/CS на каждом replan даёт большой suite-SR, хотя per-chunk isolate headroom мал.

*Demote:* длинные oracle-ветки (edge, GATE A, value(k,R) grid) — **Appendix A «Extended counterfactual study»** одним абзацем + ссылка; в основном тексте — 2–3 предложения «мы проверяли counterfactual oracle; per-chunk value ≈ 0; см. приложение».

### 2.4. **Section 4 — Consensus selection (CS) and scaling (PDF core, обновить формулировки)**

- Fig. 1 pipeline; Fig. 3 scaling N; Table 4 **только как «historical / multi-benchmark overview»** с оговоркой протокола (§5).
- CS-D (AWR distill) — один абзац + одна строка в таблице: deployable single-forward без N× AR at test time.
- Не строить секцию вокруг «CS vs PDF CS-8» — vote уже сверен (can 85.6 и т.д.); упомянуть в footnote или reproducibility.

### 2.5. **Section 5 — Selector ablation (NEW empirical spine для AAMAS)**

**Это главная «свежая» секция после диагноза.** Структура внутри секции:

1. **LIBERO-LONG (primary ablation table):** одна компактная таблица: vote, medoid, kdpe, max_ll, base, random — suite mean ± **std** (как в ledger). Текст:
   - vote ≈ medoid → **consensus / geometric mode-seeking** (KDE по нормализованному чанку), не «просто medoid cheaper».
   - kdpe << vote, но kdpe > max_ll > base ≈ random → **endpoint density и likelihood не объясняют** suite gain; отсекает альтернативные объяснения CS.
   - Per-task (Appendix): где CS забирает запас (K6, LR1, LR2a, LR5); где medoid ≈ vote — не раздувать.

2. **Cross-benchmark (secondary):** отдельная подтаблица RM/MW **без смешения с PDF base** — только «our pipeline, same 50 scenes»: vote vs random/medoid/max_ll/KDPE. RoboCasa: **без KDPE (n/a 12D)**, честно — селекторы на кухне **не разделимы** (random ≈ vote); одна строка, не полстраницы.

3. **CS vs KDPE (paired, MW focus):** не «CS лучше на MetaWorld», а:
   - **Единственный устойчивый значимый кейс:** box-close ([E] Δ +10.0, p=0.005; [C] +6.8, p=0.028).
   - Остальные пары: n.s.; coffee/disassemble — KDPE слегка выше, без значимости.
   - stick-pull: jsonl rerun +5.6, p=0.12 — **не продавать** как второй win; старые p=0.005 вычеркнуть.
   - Формулировка: *«CS matches or exceeds KDPE on success rate; statistically significant advantage appears on box-close under paired tests; elsewhere differences are not significant at n=5 repeats.»*

4. **Near-ceiling (RM can/lift):** один абзац — все селекторы эквивалентны; CS не хуже; ценность CS проявляется при **headroom** (square, stick-pull, LIBERO suite).

### 2.6. Section 6 — Latency and cost (честность обязательна)

- **Table 5 = selector-only**, H100, can canonical; подпись: не V100 PDF, warmup 50, 8×20.
- Текст **двухуровневый:**
  - На шаге селектора CS ~**2×** быстрее KDPE (0.264 vs 0.535 ms @ N=8); IQR не пересекаются.
  - **Full forward** ~28 ms (LIBERO/can D=7): Δ(CS,KDPE) **~0.3 ms (~1%)**; generation ~23 ms. Явно: *«We do not claim end-to-end 2× speedup; the Pareto argument applies to the selector micro-step and to regimes where generation is amortized or cheaper.»*
- Medoid быстрее CS на selector-only, но проигрывает на headroom (square, stick-pull) — одно предложение, без дублирования Table 5.
- MW D=4: KDPE дешевле CS на selector — **не усреднять** с D=7; вынести в Appendix «Latency by action dimension».

### 2.7. Related work + Conclusion

- AAC/PACE/DEHP — R-axis **related**, не «мы опровергли весь мир»: scope LIBERO + proper random control.
- BoN / RoboMonkey — selection literature; wedge: ordered chunks + amortized vision + **mechanism ablation**.
- Conclusion: design rule + два deployable points (BoN/CS, CS-D), не «мы SOTA на всех бенчах».

---

## 3. Что в приложение (demote aggressively)

| Содержание | Куда |
|---|---|
| Extended K/R tables, entropy, predictor, agnostic mix | App. B |
| Counterfactual oracle, edge, GATE A, chunk-Q | App. C (краткий narrative + таблицы) |
| LIBERO per-task SR grid | App. D |
| CS vs KDPE full paired table (7 MW/RM tasks) | App. E (main text — summary + box-close highlight) |
| GPU2 latency all ckpt, MW D=4 KDPE | App. F |
| RoboCasa ccm discarded runs | App. G (reproducibility note) |
| Confirmatory [C] full grid | App. H **или** footnote column «seed-robust» |

**Не в приложение:** диагноз K/R (кратко в §3), LIBERO selector ladder, box-close CS vs KDPE, Table 5 canonical + one LIBERO latency paragraph.

---

## 4. [E] vs [C]: что показывать в теле статьи

**Рекомендация: основные paper numbers = [E]**, с **узкой confirmatory полосой**, не замена всего прогона.

Обоснование:
- Весь новый spine (LIBERO 6/6, ablation, latency) собран **[E]**; [C] закрыт только для CS vs KDPE на RM/MW (7 пар), не для всех селекторов.
- [C] **согласован по знаку** с [E] там, где важно: box-close значим в обоих; stick n.s. в обоих; coffee KDPE чуть выше в обоих; disassemble — знак разошёлся, оба n.s. → не опора для claims.
- AAMAS ценит воспроизводимость: `--policy_seed` + smoke — **paragraph in §5.1** + таблица [C] в приложении.

**Не писать** «results are robust across policy seeds» глобально. Допустимо: *«For paired CS vs KDPE, confirmatory runs with fixed policy seed replicate the only significant gain (box-close) and leave other tasks non-significant (Appendix H).»*

**Не смешивать** [E] и [C] в одной ячейке. Если рецензент требует один протокол — тогда **only [C] для RM/MW paired claims**, но LIBERO ablation остаётся [E] с явной меткой (отдельная колонка/звёздочка).

Exploratory для ablation **не слабость**, если протокол описан (fixed scenes, 5 repeats, stochastic policy sampling). Ограничение — межran variability (square 38.8 vs 33.2) → честно в Limitations + [C] как audit.

---

## 5. SE vs std и Table 4: единый знаменатель

**Для всего, что вы считаете сейчас (our pipeline):** везде **± std over 5 evaluation repeats**, с footnote «SE = std/√5» где нужны error bars на fig.

**Table 4 (PDF):** не «пересчитывать молча». Варианты:
1. **Предпочтительно для AAMAS:** оставить Table 4 как **«reported in prior draft (mixed uncertainty notation)»** одной таблицей, а **canonical multi-benchmark story** перенести на **новую Table 2** (vote / CS-D / scaling) + **Table 3 selector ablation (LIBERO)** — все ± std, один протокол aic4.
2. Если Table 4 must stay main: отдельный **Protocol mismatch** подраздел + пересчёт MetaWorld/RoboCasa к одному типу (std vs SEM) **только если** есть исходные per-repeat logs; RoboMimic base из n250 — **не смешивать** с our 5×50 без relabel колонок.

**Подписи:** запретить формулировку «± SE across five seeds» без расшифровки. Писать: *«± std over 5 evaluation repeats on fixed initial conditions (test_start_seed=10000)»*.

Парные CI (bootstrap по сценам) — **Appendix**; в тексте MW box-close — одна строка с CI и p.

---

## 6. RM ceiling и PDF base mismatch — отдельный § Limitations (не сноска)

Выделить **подраздел «Limitations and protocol alignment»** (≈0.4–0.5 стр.), не прятать в конец одним абзацем:

1. **RoboMimic baseline drift:** PDF base (can 76.3, lift 90.5) совпадает с legacy n250-замером; текущий eval pipeline на **тех же сценах и ckpt** даёт base ~84–88 (can), ~96 (lift). CS ≈ base на can → **selection gain на RM в нашем пайплайне — насыщение**, не опровержение CS на LIBERO/MW headroom. CS-8/16/32 в Table 4: запросить/указать, тот ли протокол, что base (открытый вопрос к соавторам) — в тексте: *«We report PDF Table 4 for continuity; paired selector comparisons use the current unified pipeline (Appendix …).»*

2. **Cross-host (ccm vs aic4):** RoboCasa — только aic4; ccm systematically low on some tasks — discarded.

3. **Statistical power:** 5 repeats → MDE ~6–8 pp; non-significant ≠ equivalence.

4. **LIBERO vs RM/MW:** один multitask suite vs per-task policies — не сравнивать абсолютные SR между сьютами как «generalization gap» без oговорки.

5. **Latency:** V100 (PDF) vs H100 (Table 5); claim 2× **only selector-only**.

Этот блок **защищает** ablation и box-close: рецензент видит, что вы не скрываете RM saturation.

---

## 7. Риски формулировок (чеклист)

- ❌ «CS beats KDPE on MetaWorld» → ✅ «significant only on box-close; elsewhere n.s.»
- ❌ «2× faster inference» → ✅ «~2× on selector step; ~1% end-to-end»
- ❌ «stick-pull p=0.005 vs KDPE» → ✅ jsonl +5.6, p=0.12
- ❌ смешивать our random/medoid with PDF CS-8 on RoboCasa
- ❌ KDPE on RoboCasa без n/a и без §7.1 rationale
- ❌ «five seeds» без расшифровки

---

## 8. Итог для программного комитета (one slide mental model)

**Story arc:** (1) Adaptive K/R fails on LIBERO with proper controls → don’t spend cheap AR budget on wrong knob. (2) Spend on selection → suite gains compound. (3) **Why CS works:** LIBERO ablation separates mode consensus from endpoint KDE and from likelihood. (4) **Honest costs:** selector Pareto vs KDPE; not wall-clock halving. (5) **Honest limits:** RM base/ceiling, protocol heterogeneity, [E]+[C] audit.

Эта дуга лучше соответствует AAMAS, чем «новый селектор победил baseline на N задачах», и опирается на зафиксированные в дампе числа без их выдумывания.

---

*Sign: composer-2.5-fast*
