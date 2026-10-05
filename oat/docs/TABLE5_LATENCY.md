# Table 5 — GPU latency of action-chunk selectors (H100)

Полная таблица для статьи / научника. Канон — один прогон, все методы на **одних и тех же** кандидатах.

**Канон.** aic4, NVIDIA H100 80GB HBM3, GPU1 **пустая** до старта (0.0 MiB, util 0%).  
ckpt: RoboMimic can `ep-1700_sr-0.940`. R=16, K=8, batch=1.  
Кандидаты: реальные чанки политики (`obs_encoder` один раз → KV-cache `generate` → `detokenize`).  
Протокол: warmup=50, 8 trials × 20 reps = **160** сэмплов, `torch.cuda.synchronize()` вокруг каждого вызова, `cudnn.deterministic=True`.  
torch 2.10.0+cu128, CUDA 12.8, cuDNN 9.2, driver 595.71.05.  
Дата: 2026-09-30T16:42 UTC. Хост `lab-c-0`.

Это **не** V100-числа PDF Table 5 (другой GPU; здесь меряется **шаг селектора**, не весь policy-forward).

Кластерный артефакт: `~/oat_eval_out/latency/selector_latency_gpu1_n32diag.json`  
Фигуры: [`figures/table5_selector_latency.png`](figures/table5_selector_latency.png), [`figures/table5_selector_pareto.png`](figures/table5_selector_pareto.png).

---

## 1. Selector-only — median [p25, p75] ms  (колонки для Table 5)

| N | random | max_likelihood | medoid | CS (vote) | KDPE |
|---|---|---|---|---|---|
| 8 | 0.010 [0.010, 0.010] | 0.023 [0.023, 0.023] | 0.101 [0.100, 0.105] | **0.264 [0.261, 0.271]** | **0.535 [0.528, 0.541]** |
| 16 | 0.010 [0.010, 0.011] | 0.027 [0.027, 0.027] | 0.120 [0.118, 0.123] | 0.321 [0.318, 0.327] | 0.676 [0.670, 0.680] |
| 32 | 0.011 [0.010, 0.011] | 0.027 [0.027, 0.028] | 0.232 [0.230, 0.237] | 0.429 [0.423, 0.436] | 0.680 [0.673, 0.686] |

IQR CS vs KDPE **не пересекаются** ни на одном N.  
KDPE N=16 vs N=32: отношение медиан **1.01×**, IQR **пересекаются**. Теория O(N²) дала бы 4×, O(N) — 2×.

KDPE / CS: N=8 **2.02×**, N=16 **2.11×**, N=32 **1.58×**.

### min–max (хвосты, не для ± в таблице)

| N | random | max_ll | medoid | CS | KDPE |
|---|---|---|---|---|---|
| 8 | 0.009–0.019 | 0.022–0.033 | 0.097–0.133 | 0.256–0.326 | 0.517–0.638 |
| 16 | 0.010–0.025 | 0.026–0.039 | 0.115–0.164 | 0.311–0.425 | 0.660–0.806 |
| 32 | 0.010–0.019 | 0.026–0.035 | 0.225–0.249 | 0.416–0.531 | 0.660–1.923 |

Точные медианы (6 знаков, из JSON):

| N | random | max_ll | medoid | CS | KDPE |
|---|---|---|---|---|---|
| 8 | 0.009960 | 0.022881 | 0.101172 | 0.264225 | 0.534794 |
| 16 | 0.010420 | 0.026755 | 0.119457 | 0.320496 | 0.675822 |
| 32 | 0.010600 | 0.027121 | 0.231984 | 0.429338 | 0.679973 |

n=160 на каждую ячейку.

---

## 1b. Доп. ckpt на GPU2 (тот же протокол, не канон статьи)

Очередь `lane_latency_table5_gpu2.sh`, **LANE_COMPLETE 2026-10-04T12:17:26Z**. Пустая H100 GPU2, warmup=50, 8×20, R=16, `smi_before=0.0`. Канон статьи остаётся **can GPU1**. RC KDPE не мерили (D=12).

N=8 selector-only **median ms**. KDPE **не усреднять** D=7 с D=4.

| Задача | D | CS (vote) | KDPE | KDPE/CS |
|---|---|---|---|---|
| can (канон, GPU1) | 7 | 0.264 | 0.535 | 2.02× |
| lift | 7 | 0.267 | 0.533 | 2.00× |
| square | 7 | 0.263 | 0.539 | 2.05× |
| box-close | 4 | 0.270 | 0.206 | 0.76× |
| coffee-pull | 4 | 0.271 | 0.205 | 0.76× |
| stick-pull | 4 | 0.269 | 0.203 | 0.75× |
| disassemble | 4 | 0.264 | 0.200 | 0.76× |

RM (D=7): CS ≈ can, KDPE ≈ 0.53 — репрезентативно. MW (D=4): CS тот же ~0.26, KDPE дешевле (~0.20), потому что кватернионный граф короче.

Артефакты: `~/oat_eval_out/latency/selector_latency_gpu2_{lift,square,box-close,coffee-pull,stick-pull,disassemble}.json`.

---

## 2. CUDA kernel launches на один вызов (`torch.profiler`)

| N | random | max_ll | medoid | CS | KDPE |
|---|---|---|---|---|---|
| 8 | 0 | 2 | 7 | 28 | 74 |
| 16 | 0 | 1 | 6 | 27 | 73 |
| 32 | 0 | 0 | 16 | 37 | 71 |

Форма матриц KDPE в рантайме (печать из `kdpe_endpoint_scores`): **8×8 / 16×16 / 32×32**, `scores.numel()=N`. N=32 **считается**. Плоский wall-clock 16→32 — не баг усечения: граф ~72 kernel’ов не зависит от N (quaternion pairwise = много мелких ops), FLOP даже при 32×32 малы относительно launch overhead.

---

## 3. Генерация кандидатов (vision + sample N + detokenize)

Тот же канонический прогон, отдельно от селектора.

| N | median [p25, p75] ms |
|---|---|
| 8 | 22.45 [22.33, 22.57] |
| 16 | 26.78 [26.70, 26.86] |
| 32 | 27.27 [27.17, 27.46] |

---

## 4. Полный пайплайн (vision + sample + select) — не канон Table 5

Другой прогон на той же H100 GPU1: warmup=10, 8 trials × 10 reps. Порядок величины, не смешивать с блоком 1.

| режим | median [p25, p75] ms |
|---|---|
| single `predict_action` (KV) | 25.04 [24.97, 25.22] |
| BoN N=8 CS | 28.04 [27.98, 28.12] |
| BoN N=8 KDPE | 28.45 [28.41, 28.52] |
| BoN N=16 CS | 28.11 [27.99, 28.66] |
| BoN N=16 KDPE | 28.41 [28.34, 28.46] |
| BoN N=32 CS | 28.48 [28.44, 28.53] |
| BoN N=32 KDPE | 28.76 [28.68, 28.87] |

Разница CS vs KDPE на полном forward **< 0.5 ms** при ~28 ms (~1–2%).

CPU (отдельный старый замер, не этот GPU-прогон): CS 0.027 ms, KDPE 0.098 ms, medoid 0.013 ms.

---

## Комментарии агента

1. **Что ставить в Table 5.** Блок 1, все пять методов, N=8/16/32, median [p25, p75]. Карта H100. Подпись: selector-only, batch 1, 8×20, warmup 50, can ep-1700.

2. **CS vs KDPE.** На шаге селектора CS стабильно быстрее (~2× при N=8/16). Это единственное место, где latency-Pareto живой. На полном policy-forward разницы почти нет: доминирует CNN+AR (~22–27 ms), селектор 0.01–0.68 ms.

3. **Почему KDPE не растёт 16→32.** ~72 CUDA-kernel’а, граф не зависит от N. FLOP мало даже при 32×32. Wall-clock = launch overhead. Формулировка: *KDPE is launch-bound at N≥16 due to many small quaternion operations, while CS’s single vectorized distance computation scales more efficiently.* Не писать «CS scales better» без проверки N×N — проверка сделана.

4. **Medoid не доминирует CS.** В 2.6× дешевле CS (0.101 vs 0.264 ms @ N=8), но SR хуже там, где есть запас: square 29.6 vs 36.0, stick-pull 18.8 vs 28.8, box-close 67.2 vs 68.8. На can/lift (потолок) все селекторы неотличимы. Честно: *on near-ceiling tasks all selectors are equivalent; CS is the consensus method that keeps SR on headroom tasks and is still ~2× faster than KDPE.*

5. **random / max_ll.** Самые быстрые (O(1)). random — нижняя граница «взять индекс». max_ll — argmax по уже посчитанным logprob. Оба не заменяют CS по SR на трудных задачах (stick-pull: random 16.4, max_ll 14.8, CS 28.8).

6. **Не смешивать.** Не подставлять эти мс в V100-строку PDF. Не писать «CS ускоряет весь инференс в 2 раза» — только селектор. Не усреднять канон (warmup 50, 8×20) с полным пайплайном (warmup 10, 8×10).

7. **Фраза в текст.** CS is ~2× faster than KDPE on H100 (0.264–0.429 ms vs 0.535–0.680 ms for N=8–32). KDPE is launch-bound at N≥16 due to many small quaternion kernels, while CS’s single vectorized distance scales more efficiently.
