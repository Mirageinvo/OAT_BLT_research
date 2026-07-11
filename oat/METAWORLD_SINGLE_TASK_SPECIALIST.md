# MetaWorld: Multitask Status + Single-Task Specialist Plan

Документ фиксирует:
- что уже сделано по `mt4` multitask;
- какие артефакты сейчас есть на кластере;
- как запускается новый single-task specialist пайплайн (как в RoboMimic);
- как называются директории/чекпоинты, чтобы не путаться.

---

## 1) Уже сделанный multitask (`mt4`) — факт

Текущая ветка: `robomimic`  
Кластерный контейнер: `oat_mipt_robomimic_askhabaliev_gs`

### Данные
- `data/metaworld/mt4_N50.zarr` (50 демо на задачу, 200 эпизодов total)

### Обучение (multitask)
- Tokenizer run: `output/20260707/124135_train_oattok_mw-mt4_N50/`
- Policy run: `output/20260708/032431_train_oatpolicy_mw-mt4_N50/`
- Policy ckpts:
  - `ep-0200_sr-0.280.ckpt`
  - `ep-0450_sr-0.280.ckpt`
  - `ep-1000_sr-0.240.ckpt`
  - `latest.ckpt`

### Eval (multitask interleaved)
- Chain5: `output/eval/metaworld_mt4_paper5_ep0450/`
- BoN: `output/eval/metaworld_mt4_bon_n8_n3/`

---

## 2) Single-task specialist: целевой протокол

Для каждой задачи (`box-close`, `coffee-pull`, `disassemble`, `stick-pull`) отдельно:
1. свой датасет `task_N50.zarr`
2. свой tokenizer (top-1 по `test_reconst_mse`)
3. своя policy (full train-time eval, без ускорений)
4. chain5 eval (5 seeds × 50 = 250 эпизодов на задачу)
5. BoN/AWR fully automated от лучшего чекпоинта

### Full train-time eval (без ускорений)
- `training.rollout_every=200`
- `task.policy.env_runner.n_test=250`
- `task.policy.env_runner.n_parallel_envs=4` (меняется override при необходимости)
- OAT8 eval: `use_k_tokens=8`, `entropy_threshold=0`

---

## 3) Датасеты: `fresh generation` vs `split`

Есть два валидных режима подготовки single-task датасетов:

### A. Fresh generation (предпочтительно для strict repro)
- Скрипт: `scripts/cluster_gen_metaworld_single_data.sh`
- Логика: отдельный запуск `gen_metaworld_data.py` для каждой задачи:
  - `--task_name box-close --num_episodes 50 --force`
  - `--task_name coffee-pull --num_episodes 50 --force`
  - `--task_name disassemble --num_episodes 50 --force`
  - `--task_name stick-pull --num_episodes 50 --force`
- Детали генератора (баги reset/success, диагностика): см. **§3.C**
- После каждой генерации сразу запускается:
  - `scripts/validate_metaworld_data.py ... --num-tasks 1`
- Плюс батч-проверка всех 4 датасетов:
  - `scripts/validate_metaworld_single_all.sh`

### B. Split из общего `mt4_N50.zarr` (быстро и детерминированно)
- Скрипт: `scripts/split_metaworld_mt4_zarr.py`
- Логика split:
  - эпизод `i` из `mt4_N50.zarr` -> задача `i % 4`
  - порядок задач фиксирован: `[box-close, coffee-pull, disassemble, stick-pull]`
- Ожидаем:
  - `200` эпизодов в source
  - `50` эпизодов в каждом из 4 target zarr

### Побайтовая верификация split (проведена)
- Проверка: для всех `200` эпизодов сравнение source->target по round-robin маппингу.
- Результат: `mismatches = 0`.
- Это означает, что split-датасеты побайтово идентичны соответствующим эпизодам source (без перемешивания).

### C. Генератор `gen_metaworld_data.py`: баги, фиксы, критерии (2026-07-10)

#### Контекст
При первом запуске `cluster_gen_metaworld_single_data.sh` (`mwst_regen_data`) box-close завис на `Episode 1, failed (success_count=0)` — 10+ retries подряд, 0/50 принятых эпизодов. Процесс не падал, но фактически зацикливался.

#### Баг 1 (главный): `MetaworldEnv.reset()` без `seed` не рандомизирует
В обёртке `oat/oat/env/metaworld/env.py` при `reset(seed=None)` каждый раз восстанавливается **один и тот же** MuJoCo snapshot (`env_init_state`), зафиксированный при создании env в `__init__`.

Следствие:
- expert детерминирован → retry без смены seed даёт **тот же исход** снова и снова;
- если init нерешаемый для oracle → бесконечный reject-loop на одном `episode_idx`.

**Важно:** голый `env.reset()` **не** эквивалентен «авто-рандомизации» в нашем пайплайне. Это отличается от типичного gym-поведения и именно поэтому split/mt4 могли «маскировать» проблему: в multitask 4 env-а создаются с разными init при конструировании, и box-close env мог получить удачный snapshot (в логе `gen_metaworld_mt4_N50.log`: 50/50 box-close, 0 explicit fails).

#### Баг 2 (исправлен ранее, откатили): слишком строгий success-фильтр
Был вариант: принимать эпизод только если `info["success"] == True` на **последнем** шаге (`terminal_success`), а не «success был хоть раз».

Диагностика показала: на **успешных** rollout'ах старый и новый критерии эквивалентны (мусор не попадал). Но для совместимости с уже собранным `mt4_N50.zarr` оставлен критерий mt4:

```python
if episode_success_count == 0:
    continue  # reject
```

Т.е. любой шаг с `info["success"] == True` в rollout → эпизод принят.

#### Баг 3: `wait_user_input()` при рестарте на кластере
Если zarr-директория уже существует, интерактивный prompt `[y/n]` в tmux может повиснуть. Добавлен флаг `--force` (используется в `cluster_gen_metaworld_single_data.sh`).

#### Фикс (текущая логика в `scripts/gen_metaworld_data.py`)
Цикл: `while episode_idx < total_episodes` — при fail **не** инкрементируем `episode_idx`, только `attempt_idx`.

```python
roll_seed = episode_idx * 1_000_000 + attempt_idx
obs_dict, _ = env.reset(seed=roll_seed)

# ... rollout ...

if episode_success_count == 0:
    attempt_idx += 1
    continue

attempt_idx = 0
episode_idx += 1
```

**Ловушка с жёстким seed (предупреждение):** если при retry вызывать `reset(seed=fixed)` без инкремента — снова тот же нерешаемый init → зависание. У нас `attempt_idx` растёт на каждом reject → seed меняется → следующая инициализация другая.

Oracle MetaWorld не 100% с любого init — единичный reject нормален; цикл должен перебрать другой seed.

#### Диагностика expert (кластер, box-close)
| Тест | Условия | Результат |
|------|---------|-----------|
| Fast | 1 камера, 10 rollout, `reset()` без seed | 10/10 success (old & terminal) |
| Full | 4 камеры (как gen), 20 rollout, `reset()` без seed | **19/20** success; критерии совпадают на успехах |

Вывод: oracle рабочий; 1 fail — нормальный reject на невыигрышном init; фикс с `reset(seed=...)` + `attempt_idx` обязателен для single-task gen.

#### Кластерный запуск regen
```bash
tmux new-session -d -s mwst_regen_data bash scripts/cluster_gen_metaworld_single_data.sh
# лог: logs/metaworld_single_data_regen.log
```

После фикса (2026-07-10): box-close пошёл `Episode 1/50, 2/50, ...` с `reward=1.0, success_count=1`.

#### Прочее
- zarr пишется **в конце** после всех 50 эпизодов → при kill процесса директория может остаться пустой (это не corrupt zarr, просто незавершённая запись).
- 4 RGB камеры на шаг → ~1–2 мин на эпизод; прогресс в логе медленный, это нормально.
- Warning `Constant(s) may be too high` от metaworld policies — безобидный.

---

## 4) Новые скрипты (single-task)

### Данные
- `scripts/split_metaworld_mt4_zarr.py`
  - делит `mt4_N50.zarr` в:
    - `data/metaworld/box-close_N50.zarr`
    - `data/metaworld/coffee-pull_N50.zarr`
    - `data/metaworld/disassemble_N50.zarr`
    - `data/metaworld/stick-pull_N50.zarr`

### Train
- `scripts/cluster_tokenizer_metaworld_single.sh`
  - single-task tokenizer, top-1
- `scripts/cluster_policy_metaworld_single.sh`
  - single-task policy, full промежуточный eval

### Eval / selection
- `scripts/cluster_eval_metaworld_single_chain5.sh`
  - chain5 для одной задачи (summary.json с mean/stderr)
- `scripts/select_best_ckpt_by_name.py`
  - выбор лучших ckpt по метрике из имени (`mse`/`sr`)

### BoN/AWR
- `scripts/cluster_metaworld_single_bon_awr.sh`
  - STEP1 BoN -> STEP2 collect -> STEP3 validate -> STEP4 train AWR -> STEP5 eval AWR

### Full automation
- `scripts/cluster_metaworld_single_full_pipeline.sh`
  - полностью: data -> tok -> policy -> chain5(top3->best) -> bon/awr
- `scripts/cluster_launch_metaworld_single_4.sh`
  - запуск 4 задач параллельно через tmux
- `scripts/cluster_gen_metaworld_single_data.sh`
  - чистая последовательная генерация 4 отдельных single-task датасетов
- `scripts/validate_metaworld_single_all.sh`
  - батч-валидация всех 4 single-task zarr

---

## 5) Нейминг артефактов (single-task)

Используется run-tag:
- `mwst_<task>_<YYYYMMDD>_<HHMMSS>`

Примеры:
- Лог полного пайплайна:
  - `logs/mwst_box-close_20260710_201500_full_pipeline.log`
- Eval chain5:
  - `output/eval/metaworld_box-close_paper5_ep-0400_sr-0.520_mwst_box-close_20260710_201500/`
- AWR dataset:
  - `my_datasets/awr_mw_box-close_st_20260710_201500.npz`
- AWR ckpt:
  - `my_models/policy_awr_mw_box-close_st_20260710_201500.ckpt`

Train run dirs:
- tokenizer: `output/<date>/<time>_train_oattok_mw-<task>_st_N50/`
- policy: `output/<date>/<time>_train_oatpolicy_mw-<task>_st_N50/`

---

## 6) Запуск 4 параллельных run’ов (не трогая существующие)

Скрипт:
- `bash scripts/cluster_launch_metaworld_single_4.sh`

Что делает:
- создаёт tmux-сессии:
  - `mwst_box_close`
  - `mwst_coffee_pull`
  - `mwst_disassemble`
  - `mwst_stick_pull`
- GPU mapping по умолчанию:
  - box-close -> GPU0
  - coffee-pull -> GPU1
  - disassemble -> GPU0
  - stick-pull -> GPU1
- если сессия уже существует, задача **пропускается**, ничего не убивается

---

## 7) TODO checklist

- [x] Зафиксировать multitask артефакты (пути и результаты)
- [x] Подготовить split `mt4_N50.zarr` -> 4 single-task zarr
- [x] Подготовить single-task train скрипты (tokenizer/policy)
- [x] Подготовить single-task chain5 eval
- [x] Подготовить single-task BoN/AWR pipeline
- [x] Подготовить full automation script per task
- [x] Подготовить launcher на 4 параллельных tmux с безопасным неймингом
- [x] Зафиксировать баги/фиксы `gen_metaworld_data.py` (reset seed, success criterion, `--force`)
- [ ] Дождаться `cluster_gen_metaworld_single_data.sh` (4 zarr + validate)
- [ ] Запустить 4 single-task пайплайна на кластере
- [ ] Мониторить логи и собрать таблицу per-task (base/BoN/AWR)
