# LIBERO-LONG — результаты селекторов (CS / KDPE / baselines)

Состояние на **6 Oct 2026, 09:50 MSK**. Хост aic4 (`lab-c-0`). Метка **[E]** = paper (без `--policy_seed`). Не смешивать с **[C]** (`--policy_seed 0`).

Один скаляр в статью: `mean_success_rate_mean` ± `mean_success_rate_std` (n=5 экспа).  
Полный разбор латентности селектора: [`TABLE5_LATENCY.md`](TABLE5_LATENCY.md).  
Таблица CS vs KDPE по RM/MW: [`CS_VS_KDPE_TABLE.md`](CS_VS_KDPE_TABLE.md).

---

## 0. Репрезентативность латентности LIBERO vs Table 5 — вердикт

**LIBERO ckpt замерен** 6 Oct 06:47 UTC, пустая GPU1 (smi_before **0.0 MiB**), тот же протокол Table 5 (warmup=50, 8×20, `oat_code_seed`): `~/oat_eval_out/latency/selector_latency_gpu1_libero10.json`.

Два часов — не смешивать:

| Часы | Что | LIBERO N=8 | can D=7 (replicate) |
|---|---|---|---|
| **Selector-only** | индекс по готовому `[N,R,7]` | CS **0.272** / KDPE **0.553** (~2.0×) | 0.268 / 0.552 |
| **Generation** | CNN + AR×N + detokenize | **22.73** ms | 22.68 |
| **Full forward** | gen + select | CS **27.76** / KDPE **28.09** (Δ +0.33) | 27.64 / 27.89 |
| single KV | без BoN | **24.78** | 24.91 |

Совпадает с can/lift (D=7). Table 5 selector-only канон (can `n32diag` 0.264 / 0.535) остаётся колонками статьи; LIBERO — подтверждение, не замена.

MW D=4: CS тот же ~0.27, **KDPE ~0.20 — не LIBERO**. Full MW ~31–33 ms (другой gen).

Итог: *на селекторе CS ≈ 2× KDPE (0.27 vs 0.55 ms). End-to-end оба ~28 ms; селектор <2% forward.*

---

## 1. Протокол SR

| | |
|---|---|
| ckpt | `~/cs_libero_hf/policy_ep-0250_sr-0.596.ckpt` (HF `Mirageinv/CS-libero`) |
| код eval | `~/oat_code_kdpe` (`cell_libero.sh`) |
| suite | LIBERO-LONG / `libero10`, 10 задач |
| n_test | **500** (по 50 инициализаций на задачу) |
| num_exp | **5** |
| test_start_seed | **10000** |
| n_parallel_envs | **4** |
| K / T / topk | 8 / 1.0 / 10 |
| `--entropy_threshold` | 0 (полный бюджет) |
| N (BoN) | 8; **base** — `bon_free=0` (один sample) |
| `--selector_seed` | 0 (в `cell_libero.sh` на все BoN-методы; это RNG random-селектора, **не** `--policy_seed`) |
| KDPE | `--kdpe_bandwidth 0.05` |
| симулятор | OSMesa, `~/gl-prefix` |
| метка | **[E]**, без `--policy_seed` |

Рестарт оставшихся четырёх ячеек: **2026-10-05T14:03:05Z**. json готовы **2026-10-06 04:37–05:08 UTC**. vote/random сняты раньше (2 Oct) тем же скриптом — json не перегоняли (`SKIP` если `eval_log.json` есть).

---

## 2. Suite SR (скаляр в статью)

± = **std** по 5 экспам. SE = std/√5.

| Селектор | `mean_success_rate_mean` | std | SE | vs CS | json mtime UTC |
|---|---|---|---|---|---|
| **vote (CS)** | **0.6652** | 0.0250 | 0.0112 | — | 2 Oct 05:18 |
| medoid | 0.6628 | 0.0280 | 0.0125 | −0.002 | 6 Oct 04:37 |
| kdpe | 0.6048 | 0.0094 | 0.0042 | −0.060 | 6 Oct 04:53 |
| max_likelihood | 0.5840 | 0.0103 | 0.0046 | −0.081 | 6 Oct 05:08 |
| base (без BoN) | 0.5580 | 0.0176 | 0.0079 | −0.107 | 6 Oct 05:05 |
| random | 0.5540 | 0.0152 | 0.0068 | −0.111 | 2 Oct 09:44 |

Все: `num_exp=5`, `mean_tokens_used=8.0`.

Per-exp suite (только vote / random были выписаны раньше из логов; в json — агрегат):

- vote: 0.672 / 0.660 / 0.704 / 0.652 / 0.638
- random: 0.558 / 0.568 / 0.558 / 0.528 / 0.558

Читать: на LIBERO **CS ≈ medoid** (разница внутри шума). KDPE ниже CS на ~6 pp. random ≈ base. Это **не** потолок can/lift: запас по SR есть, и consensus (vote) его забирает, в отличие от max_ll/random.

---

## 3. Per-task SR

Каждая клетка: mean ± std по 5 экспам (те же 50 init/задача × 5). Имена — официальные `task_name` из `eval_log.json`.

Сокращения: K3 плита+мока · K4 миска в ящик · K6 кружка в микроволновку · K8 две моки · LR1 суп+cream cheese · LR2a суп+томат · LR2b cream cheese+масло · LR5 две кружки на тарелки · LR6 кружка+пудинг · S1 книга в лоток.

| Задача | vote | medoid | kdpe | max_ll | base | random |
|---|---|---|---|---|---|---|
| K3 `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it` | **0.928±0.030** | 0.944±0.043 | 0.864±0.067 | 0.896±0.030 | 0.844±0.036 | 0.796±0.043 |
| K4 `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it` | 0.952±0.030 | **0.968±0.018** | 0.924±0.033 | 0.932±0.046 | 0.904±0.017 | 0.880±0.042 |
| K6 `KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it` | **0.908±0.033** | **0.908±0.036** | 0.752±0.054 | 0.784±0.048 | 0.768±0.077 | 0.808±0.018 |
| K8 `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove` | 0.640±0.110 | **0.696±0.087** | 0.584±0.107 | 0.544±0.099 | 0.512±0.070 | 0.476±0.082 |
| LR1 `LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket` | **0.608±0.076** | 0.572±0.076 | 0.508±0.064 | 0.472±0.036 | 0.452±0.064 | 0.472±0.041 |
| LR2a `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket` | **0.540±0.071** | 0.520±0.080 | 0.464±0.083 | 0.460±0.028 | 0.420±0.032 | 0.372±0.046 |
| LR2b `LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket` | 0.516±0.041 | **0.580±0.037** | 0.564±0.061 | 0.440±0.051 | 0.436±0.048 | 0.436±0.036 |
| LR5 `LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate` | **0.652±0.054** | 0.580±0.072 | 0.520±0.077 | 0.528±0.044 | 0.456±0.041 | 0.504±0.118 |
| LR6 `LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate` | **0.432±0.046** | 0.428±0.103 | 0.420±0.066 | 0.380±0.075 | 0.372±0.090 | 0.360±0.040 |
| S1 `STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy` | **0.476±0.062** | 0.432±0.048 | 0.448±0.091 | 0.404±0.065 | 0.416±0.054 | 0.436±0.079 |

SE по задачам лежат в тех же json (`…/mean_success_rate_stderr`).

Где CS забирает запас: K6, LR1, LR2a, LR5 (vote заметно выше base/random). K8 шумный (std ~0.09–0.11). Medoid иногда выше vote (K4, K8, LR2b) — suite в сумме всё равно ничья.

---

## 4. Латентность (замер на LIBERO ckpt + Table 5)

Протокол: aic4 GPU1 пустая, warmup=50, 8×20, Ns=8/16/32, batch 1, реальные кандидаты, `oat_code_seed`.

**Selector-only** = выбор индекса по уже сгенерированному чанку (Table 5).  
**Full forward** = `predict_action_bon_free` = vision + sample N + detokenize + select.  
**Generation** = full минус select.

### 4.1 LIBERO `policy_ep-0250`, пустая GPU1, 6 Oct 06:47 UTC

Артефакт: `~/oat_eval_out/latency/selector_latency_gpu1_libero10.json`.

| N | selector CS | selector KDPE | KDPE/CS | gen | full CS | full KDPE |
|---|---|---|---|---|---|---|
| 8 | **0.272 [0.269, 0.279]** | **0.553 [0.549, 0.559]** | 2.03× | 22.73 [22.67, 22.86] | **27.76 [27.67, 27.82]** | 28.09 [28.02, 28.16] |
| 16 | 0.329 [0.325, 0.336] | 0.689 [0.684, 0.694] | 2.09× | 26.98 | 27.31 | 27.88 |
| 32 | 0.439 [0.432, 0.453] | 0.696 [0.689, 0.715] | 1.59× | 27.57 | 27.80 | 28.28 |

single KV: **24.78 [24.67, 24.87]** ms. IQR selector CS vs KDPE не пересекаются.

KDPE verify N=8/16/32: матрицы `(8,8)` / `(16,16)` / `(32,32)`, `LAST_CALL` есть.

### 4.2 Рядом с каноном can (Table 5 блок 1) и replicate

| | can канон `n32diag` | can replicate 6 Oct | LIBERO 6 Oct |
|---|---|---|---|
| selector N=8 CS / KDPE | **0.264 / 0.535** | 0.268 / 0.552 | 0.272 / 0.553 |
| full N=8 CS / KDPE | нет в `n32diag` | 27.64 / 27.89 | 27.76 / 28.09 |

Колонки Table 5 = can канон selector-only. LIBERO/replicate — подтверждение D=7.

### 4.3 MW D=4 (не LIBERO)

Selector CS ~0.26–0.27 (как D=7). KDPE ~0.20. Full ~31–33 ms. Не усреднять KDPE с D=7.

---

## 5. Артефакты (кластер aic4)

### 5.1 Success rate

Корень: `~/oat_eval_out/paired/libero10/`

| Метод | `eval_log.json` | лог | json UTC |
|---|---|---|---|
| vote | `…/libero10/vote/eval_log.json` | `…/libero10/vote.log` | 2026-10-02 05:18 |
| random | `…/libero10/random/eval_log.json` | `…/libero10/random.log` | 2026-10-02 09:44 |
| medoid | `…/libero10/medoid/eval_log.json` | `…/libero10/medoid.log` | 2026-10-06 04:37 |
| kdpe | `…/libero10/kdpe/eval_log.json` | `…/libero10/kdpe.log` | 2026-10-06 04:53 |
| base | `…/libero10/base/eval_log.json` | `…/libero10/base.log` | 2026-10-06 05:05 |
| max_likelihood | `…/libero10/max_likelihood/eval_log.json` | `…/libero10/max_likelihood.log` | 2026-10-06 05:08 |

Ключи json: `mean_success_rate_{mean,std,stderr}`, `num_exp`, `checkpoint`, `bon_config`, плюс на каждую из 10 задач `TASK/mean_success_rate_{mean,std,stderr}`.

ckpt: `~/cs_libero_hf/policy_ep-0250_sr-0.596.ckpt`  
скрипт: `~/queues/cell_libero.sh`  
HF моделей: `Mirageinv/CS-libero`

### 5.2 Latency

| Файл | Что |
|---|---|
| `~/oat_eval_out/latency/selector_latency_gpu1_libero10.json` (+ `.md`) | **LIBERO** selector+gen+full, GPU1 empty, 2026-10-06 06:50 UTC |
| `~/oat_eval_out/latency/selector_latency_gpu1_n32diag.json` (+ `.md`) | **канон Table 5** selector+gen (без full), can, 2026-09-30 |
| `~/oat_eval_out/latency/selector_latency_gpu1_clean_20261006.json` (+ `.md`) | can replicate с full, GPU1 empty, 2026-10-06 06:31 |
| `~/oat_eval_out/latency/selector_latency_gpu2_{lift,square,box-close,coffee-pull,stick-pull,disassemble}.json` | доп. ckpt, GPU2, LANE_COMPLETE 2026-10-04 (full уже был) |

Код замера: `~/oat_code_seed/scripts/measure_selector_latency_gpu.py` (нужен `LAST_CALL` в `kdpe.py`; **не** `oat_code_kdpe`).

---

## 6. Что не делать при выгрузке / в тексте

- Не подставлять MW KDPE (D=4, ~0.20 ms) как latency LIBERO.
- Не писать «CS ускоряет инференс в 2 раза» — только selector-only; full ~28 ms, Δ~0.3 ms.
- Не смешивать [E] с `--policy_seed 0`.
- Не перегонять vote/random: json уже paper-числа.
- Не цитировать warmup-10 full (~28 ms GPU1.json) вместо 8×20.
- Канон Table 5 selector = `n32diag`; LIBERO full/selector = `libero10.json`.
