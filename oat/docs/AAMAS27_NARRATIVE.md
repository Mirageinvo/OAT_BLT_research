# AAMAS — нарратив статьи (синтез совета моделей)

**Дата:** 2026-10-06. **Venue:** AAMAS (не AAAI).  
**Скаляры:** только **[E]** (без `--policy_seed`). **[C]** — аудит знака, в средние не входит.  
**Источники чисел:** `AAMAS27_DUMP.md`, `LIBERO_LONG_REZULTATY.md`, `CS_VS_KDPE_TABLE.md`, `TABLE5_LATENCY.md`, `SELECTOR_BASELINES_STATUS.md`. PDF в репо нет; претензии PDF — из дампа.

**Совет.** Бесплатные модели из auto. Gemini Flash и Muse упёрлись в лимит Other Models — не голосовали. Состав:

| Файл | Модель | Роль |
|------|--------|------|
| [`council/grok.md`](council/grok.md) | grok-4.7-high-fast | substantives + запреты |
| [`council/composer.md`](council/composer.md) | composer-2.5-fast | структура 8 стр. |
| [`council/chair_skeptic.md`](council/chair_skeptic.md) | grok-4.6 | kill-shots, сужение тезиса |

Ниже — **согласованный вердикт председателя** (этот файл). Где совет расходится — помечено.

Артефакты: git-ветка `aamas27_selector_baselines`; HF [`hackhackhack66666/aaai27-models`](https://huggingface.co/hackhackhack66666/aaai27-models) → `eval/aamas27_selector/`, `docs/aamas27/`.

---

## 0. One-sentence pitch (допустимый)

На каждом перепланировании замороженная action-chunking политика обязана выбрать один из N сэмплированных чанков. На LIBERO-LONG правило **консенсуса по траектории чанка** (vote / medoid) ранжируется выше endpoint-KDE, likelihood и random; адаптация бюджета K/R при matched cost (PDF Tables 1–3) этого не даёт. Это **ранжирование правил выбора на одном длинном сьюте**, не method-win CS и не 2× ускорение инференса.

Скептик: если это не влезает без раздувания — workshop / short note, не полный method paper.

---

## 1. Что совет единогласно поддерживает

1. **Lead = диагноз + ablation, не «CS победил KDPE».** Adaptive K/R (PDF) — фон. Свежий эмпирик — линейка селекторов на LIBERO-LONG **[E]** (`n_test=500`, n=5):

   | правило | SR |
   |---------|-----|
   | vote (CS) | **0.665±0.025** |
   | medoid | **0.663±0.028** |
   | KDPE | 0.605±0.009 |
   | max_likelihood | 0.584±0.010 |
   | base | 0.558±0.018 |
   | random | 0.554±0.015 |

   Random ≈ base → сам N не помогает. Likelihood не объясняет зазор до консенсуса. Endpoint-ядро (KDPE, индекс R−1, bandwidth 0.05) до консенсуса на этом сьюте не дотягивает (~6 п.п. по средним).

2. **CS vs KDPE вне LIBERO сужать до одной ячейки.** Значим только **box-close**: [E] +10.0, CI [+3.6, +16.8], p=0.005; [C] +6.8, p=0.028. Stick jsonl +5.6, p=0.12 — не второй win. Coffee [E] −2.8, p=0.25 (KDPE чуть выше). KDPE значимо лучше CS нигде нет. Старые stick 18.0 p=0.005 и coffee 41.6 без jsonl **запрещены**.

3. **Два часов латентности.** Table 5 = selector-only H100 can: N=8 CS **0.264** vs KDPE **0.535** мс (~2.02×). Full forward D=7 оба ~**28** мс, Δ ~0.3 мс. LIBERO 0.272 / 0.553 подтверждает D=7, не заменяет канон. MW D=4: KDPE ~0.20, не усреднять с D=7. RC KDPE n/a (12D).

4. **Не смешивать пайплайны.** [E]≠[C]. PDF Table 4 RoboMimic base = n250, не наш 5×50. Наши random/medoid не сравнивать с PDF CS-8. ccm RC (кроме close-drawer) discarded. ± нашего протокола = **std** по 5 повторам (сноска SE=std/√5).

5. **RM в текущем пайплайне — потолок, не витрина.** can PDF base 76.3 vs наши 84–88; CS-8 PDF 85.6 = наш vote 85.6; парный CS−base +1.2, p=0.78. Это **Limitations**, не сноска. Причина сдвига не найдена.

---

## 2. Где совет расходится (председательное решение)

| Тема | grok / composer | скептик | Решение для текста |
|------|-----------------|---------|-------------------|
| Vote vs medoid на LIBERO | «семейство консенсуса / mode-seeking» | метод не идентифицирован; CS ≠ уникальный | Писать **consensus family {vote, medoid}**, не «CS уникален». KDE-vote как *реализация*, не как доказанный механизм |
| «Порядок однозначен» на LIBERO | да | нет парного p | Порядок *средних* — да. «Значимо лучше KDPE на LIBERO» — **нет** |
| «CS matches or exceeds KDPE» | composer | ложно: coffee/lift/disassemble точечно KDPE ≥ CS | Только: *significant advantage only on box-close; elsewhere n.s.* |
| Headroom square/stick | medoid не доминирует CS (36.0 vs 29.6; 28.8 vs 18.8) | нет парного p | Приложение / одно предложение, не claim |
| K/R как lead | composer: §3 до CS | ок только если комитет принимает PDF 1–3 | Оставить короткий §3 из PDF, без новых oracle/edge чисел |
| Compounding / isolate / GATE | composer тянет в мотивацию | не в дампе overlay | 2–3 предложения + appendix, не вклад AAMAS overlay |
| CS-D / N-scaling Table 4 | composer: секция 4 | не пересчитаны в текущем пайплайне | Historical Table 4 с оговоркой протокола, не «свежий» результат |

---

## 3. Запрещённые формулировки (чеклист в PDF)

- «CS ускоряет инференс в 2 раза» / Pareto времени реакции.
- «CS значимо лучше KDPE на MetaWorld / на двух задачах».
- Смешение [E] и [C]; «robust across policy seeds» глобально (disassemble знак разошёлся).
- PDF can +9 п.п. как результат *текущего* пайплайна.
- Усреднение KDPE D=4 с D=7; KDPE на RoboCasa без n/a.
- «five seeds» = обученные модели; RM/MW = «one multi-task policy».
- ccm coffee/microwave/faucet как paper.
- Парный p на LIBERO (его нет).
- warmup-10 latency; V100 PDF Table 5 в одну строку с H100.

---

## 4. Рекомендуемый скелет (8 стр. + приложение)

Согласован grok + composer, сужен скептиком.

1. **Агент и обязательство** (~0.7 стр.). Один проход восприятия, N чанков, одно действие до следующего replan. Вопрос: правило выбора, не новое обучение.
2. **Диагностика K/R** (~0.8 стр.). PDF Tables 1–3 сжато. Не «опровергли AAC/PACE», а: на LIBERO/OAT с random-контролем adaptive K/R не бьёт matched-cost.
3. **Правила выбора.** CS=KDE-мода по чанку; medoid; KDPE endpoint; max_ll; random. KDPE только D∈{4,7}.
4. **Главная таблица — LIBERO-LONG [E]** (suite ± std). Per-task — приложение. Не достраивать p.
5. **CS vs KDPE, 7 пар [E].** Жирный только box-close. [C] — приложение, одна фраза в тексте про совпадение знака на box-close.
6. **Где запаса нет.** RM saturation; RC неразделимость; microwave vote 46.4±4.6 vs PDF.
7. **Цена шага.** Table 5 selector-only; e2e ~28 мс абзацем.
8. **Limitations** (~0.5 стр., не сноска). RM base drift; SE/std в Table 4; [E] межran (square 38.8 vs 33.2); MDE ~6–8 п.п.; novelty vs RoboMonkey/MG-Select = ablation механизма + K/R-диагноз, не «первый BoN».
9. **Conclusion.** Design rule: extra test-time compute → selection, не adaptive K/R. Deployable: CS at N=8; CS-D одной строкой если оставляем Table 4.

**Приложение:** per-task LIBERO; полные paired CI; [C] сетка; GPU2 / D=4 latency; ccm discarded; oracle/edge если вообще оставлять.

**Не в main:** «CS > medoid на LIBERO»; N-scaling как новый вклад без пересчёта; value-Q / AWR16 как часть *этого* overlay.

---

## 5. Как подать сьюты (чтобы не смешать)

| Сьют | Роль в AAMAS | Что показывать | Чего не делать |
|------|----------------|----------------|----------------|
| **LIBERO-LONG** | primary ablation | 6 селекторов, один скаляр `mean_success_rate_mean` | выдумывать p; звать CS уникальным vs medoid |
| **MetaWorld** | CS vs KDPE | box-close + таблица n.s. | «победа на MW» |
| **RoboMimic** | saturation / protocol | наш base vs CS; PDF отдельно | Table 4 +9 п.п. как текущий gain |
| **RoboCasa** | negative / n/a KDPE | aic4; селекторы ≈; 12D | ccm; сравнивать random с PDF CS-8 |
| **Latency** | micro-step Pareto | can канон + LIBERO confirm D=7 | 2× e2e; pool D=4 |

---

## 6. Атаки рецензента и заготовки ответа

| Атака | Ответ |
|-------|--------|
| Это просто RoboMonkey/BoN | Ablation: тот же N, random≈base, max_ll≈base+2 п.п.; выигрыш у consensus family. Wedge = ordered chunks + amortized vision + K/R-negative. Не «первый BoN» |
| Vote = medoid, где метод? | Согласны: на suite не отделяем. CS — удобная реализация консенсуса; medoid дешевле на селекторе (0.101 vs 0.264 мс) и на LIBERO не хуже. Честно в тексте |
| Одна значимая ячейка из семи | Да. Multiple comparisons. Claim = box-close only |
| PDF can 76→86 vs ваш 88≈86 | Limitations: другой eval (n250 vs 5×50). Selection gain на RM в *этом* пайплайне нет |
| RC CS ниже PDF | Да, 3–6 п.п.; селекторы неразделимы. Не витрина |
| 2× latency | Только selector-only D=7. E2e ~1% |
| Unseeded [E] | Протокол описан; [C] audit на 7 парах; square межran в Limitations |
| KDPE слабый baseline | Offline: KDPE≠vote ~75–78%; на LIBERO 0.605 > max_ll. Не strawman. На D=4 дешевле CS |

---

## 7. Что чинить в тексте PDF до сабмита

1. Подпись неопределённости: везде std нашего протокола; Table 4 — «historical, mixed SE/std».
2. «Five seeds» / «one multi-task policy» — переписать по сьютам (`AAMAS27_DUMP.md` §4.2).
3. Table 5: заменить/дополнить H100 selector-only; V100 оставить как «prior GPU» или убрать.
4. Не тащить в abstract RM +9 п.п.
5. KDPE: D=4/7; RoboCasa n/a.
6. Exploratory vs confirmatory: paper = [E]; [C] = appendix.

Открытые вопросы научнику (из дампа, совет не закрывает): тот ли n250 у CS-8/16/32 RoboMimic, что у base; добивать ли 10 повторов на n.s. парах (скептик: нет, MDE всё ещё ~5–6 п.п.).

---

## 8. Итог совета одной карточкой

**Делать:** LIBERO ranking {vote ≈ medoid} > KDPE > max_ll ≈ base ≈ random + box-close + честный K/R-фон + честная латентность + толстый Limitations.

**Не делать:** method-win CS, multi-suite law, 2× wall-clock, PDF RM gain в текущем пайплайне, «robust seeds», KDE как идентифицированный механизм.

model=chair-synthesis (grok-4.6) · votes: grok-4.7, composer-2.5, grok-4.6-skeptic
