# OAT на RoboCasa: пошаговый план (порт → обучение → BoN → AWR)

Цель — воспроизвести на **RoboCasa** два наших позитива с LIBERO: **verifier-free Best-of-N**
(инференс-отбор, +0.11 SR) и **AWR-дистилляцию** (запекание отбора в один проход, +0.10 SR).
RoboCasa выбран, т.к. он **на robosuite/MuJoCo — как LIBERO** → максимум реюза существующего кода.

> **3 факта наверх (сэкономят недели):**
> 1. OAT сейчас — только LIBERO. RoboCasa надо **сначала портировать** (Фаза 1), но т.к. это
>    robosuite — порт = адаптация LIBERO-файлов, не с нуля. Механика порта общая с
>    `ROBOMIMIC_BON_AWR_GUIDE.md` (можно читать вместе).
> 2. **BoN обучения НЕ требует** (инференс-флаги), **AWR требует** (сбор + `train_awr.py`). Общий
>    предпосыл — одна обученная OAT-RoboCasa политика.
> 3. **3 решения принять ДО кода** (они определяют весь порт) — см. Фазу 0.

---

## Фаза 0 — Решения до старта (критично)

**Р1. Одна простая fixed-base атомарная задача для начала.** НЕ mobile-manipulation, НЕ composite.
Бери механически простую (больше headroom для BoN): напр. `OpenDrawer` / `CloseDrawer` /
`TurnOnSinkFaucet` / `OpenDoor`. Избегай PnP-задач на старте (сложнее → низкий SR → нет headroom).
*Проверь*: чтобы у выбранной задачи база НЕ двигалась (иначе экшн > 7D, ломает токенизатор).

**Р2. 2 камеры (как OAT).** RoboCasa даёт 3+ (`robot0_agentview_left`, `robot0_agentview_right`,
`robot0_eye_in_hand`). Маппим **2**: `robot0_agentview_left → agentview_rgb`,
`robot0_eye_in_hand → robot0_eye_in_hand_rgb`. (Расширять до 3 = менять shape_meta + энкодер, не надо.)

**Р3. Экшн = 7D (arm OSC_POSE 6 + gripper 1), base-DoF ОТБРОСИТЬ.** OAT-токенизатор жёстко на 7D
(3 pos + 3 rot + 1 gripper). Настрой robosuite-контроллер на fixed-base OSC_POSE и бери только
эти 7 измерений из демо/действий. **Если экшн окажется >7 — это стоп-фактор, решай тут.**

Ориентир по формату (существующий LIBERO-конфиг, `oat/config/task/policy/libero/libero10.yaml`):
```yaml
shape_meta:
  obs:
    agentview_rgb:          {shape: [128,128,3], type: rgb}
    robot0_eye_in_hand_rgb: {shape: [128,128,3], type: rgb}
    robot0_eef_pos:         {shape: [3], type: state}
    robot0_eef_quat:        {shape: [4], type: state}
    robot0_gripper_qpos:    {shape: [2], type: state}
    task_uid:               {shape: [1], type: state}
  action: {shape: [7]}
```
RoboCasa даёт `robot0_eef_pos/quat/gripper_qpos` теми же robosuite-ключами → **state-порт совпадает**.
Меняются только: пути камер (Р2), экшн-ключ (Р3), `task_uid` (для одной задачи = константа 0).

---

## Фаза 1 — Установка RoboCasa

```bash
cd oat
# RoboCasa + его форк robosuite (версии критичны — ставь по README RoboCasa):
git clone https://github.com/robocasa/robocasa && cd robocasa && pip install -e . && cd ..
# их robosuite-форк (RoboCasa требует конкретную ветку!):
git clone https://github.com/ARISE-Initiative/robosuite -b robocasa_v0.1 && pip install -e robosuite
python robocasa/robocasa/scripts/download_kitchen_assets.py     # ассеты кухонь (обяз.)
python robocasa/robocasa/scripts/setup_macros.py                # макросы
```
⚠️ Версии robosuite/mujoco RoboCasa могут конфликтовать с OAT/LIBERO. Ставь **в отдельное uv/conda
окружение** или проверь совместимость. Это первый риск — реши до порта.

Sanity RoboCasa (их родной eval работает?):
```bash
python robocasa/robocasa/demos/demo_random_action.py --task <твоя_задача>
```

---

## Фаза 2 — Порт OAT → RoboCasa (инженерия, один раз)

Зеркалим LIBERO-файлы. Интерфейс, который дёргают раннер/collect (из `oat/env/libero/env.py`):
`reset()`, `step(action)→(obs,reward,done,False,info)`, атрибуты `done`, `cur_step`,
`max_episode_steps`, метод `_extract_obs(raw)` (в формат политики), `render()`, `close()`.

**2A. Env-обёртка** — `oat/env/robocasa/env.py::RoboCasaEnv`.
- Ориентир: `oat/env/libero/env.py::LiberoEnv`. Внутри строим robosuite/RoboCasa-env из
  `robocasa.utils.env_utils` (или `robosuite.make` с kwargs задачи). Контроллер — OSC_POSE fixed-base
  (Р3). `_extract_obs`: вытащить 2 камеры (Р2) + eef_pos/quat/gripper_qpos, отдать dict как LIBERO.
- Реализуй ровно интерфейс выше. (`get_sim_state/set_state` НЕ нужны для BoN/AWR — только для
  counterfactual-оракула, его тут не мерим.)

**2B. Factory** — `oat/env/robocasa/factory.py::get_subtasks(task_name)` → список задач (для одной
стартовой задачи вернёт `[task]`).

**2C. Конвертер датасета** — `scripts/convert_robocasa_dataset.py` +
`oat/env/robocasa/dataset_conversion.py::convert_robocasa_hdf5_to_zarr`.
- Ориентир: `oat/env/libero/dataset_conversion.py`, `scripts/convert_libero_dataset.py`. RoboCasa
  демо — HDF5 в robomimic-формате → уложить в ту же Zarr-схему: `obs_keys` (2 rgb + 3 state + task_uid),
  `action` (7D по Р3). Смапь имена камер RoboCasa → `agentview_rgb`/`robot0_eye_in_hand_rgb`.

**2D. Env-runner** — `oat/env_runner/robocasa_runner.py::RoboCasaRunner`.
- Ориентир: `oat/env_runner/libero_runner.py::LiberoRunner`. **Копируй 1-в-1**, замени
  `LiberoEnv`→`RoboCasaEnv`, `get_subtasks`. Он уже вызывает `policy.predict_action_adaptive(...)` и
  считает `mean_success_rate`/`mean_tokens_used` → **BoN и var-R работают без правок раннера**.

**2E. Конфиги** — `oat/config/task/tokenizer/robocasa/<task>.yaml`,
`oat/config/task/policy/robocasa/<task>.yaml`.
- Ориентир: соответствующие `.../libero/libero10.yaml`. Поменять: `zarr_path`, `shape_meta`
  (камеры/экшн по Р2/Р3), `env_runner._target_ → RoboCasaRunner`, `task_name`, `max_episode_steps`
  (RoboCasa атомарные ~500). **Токенизатор: сохрани `token_dropout_mode: 'pow2'`, `num_registers: 8`.**

**2F. Адаптировать сбор AWR** — `scripts/collect_awr_dataset.py` жёстко импортит `LiberoEnv` и
`get_subtasks` (через `branch_value_k.py`: `build_obs`, `decode_k`, `env_kwargs_from_cfg`).
Параметризуй по среде (`--env robocasa`) или сделай копию под `RoboCasaEnv`. Проверь `build_obs`
(To-стэкинг/порты наблюдений) на libero-специфику и обобщи.
> `eval_policy_sim.py` порта НЕ требует — он инстанцирует раннер через Hydra из конфига → подхватит
> `RoboCasaRunner` сам. Порт нужен только для сбора AWR (2F), не для eval/BoN.

**DoD Фазы 2:** обучение (Фазы 4–5) стартует, baseline-eval (Фаза 6) даёт ненулевой SR.

---

## Фаза 3 — Данные

```bash
# скачать MimicGen-датасет задачи (бери БОЛЬШОЙ machine-generated ~3000 демо, не 50 human):
python robocasa/robocasa/scripts/download_datasets.py --tasks <task> --dataset_type mg
# конвертировать в Zarr (скрипт из 2C):
uv run python scripts/convert_robocasa_dataset.py --root_dir data/robocasa --task <task>
# проверь: zarr создан, obs/action непустые, action.shape[-1]==7
```

---

## Фаза 4 — Обучить токенизатор (VQ-VAE, замораживается для стадии 2)

```bash
MUJOCO_GL=egl uv run accelerate launch scripts/run_workspace.py \
  --config-name=train_oattok task/tokenizer=robocasa/<task>
```
Дождись плато recon-MSE (у нас на LIBERO ~0.002). Сохрани чекпоинт токенизатора → он замораживается.

## Фаза 5 — Обучить политику (frozen токенизатор → AR-голова)

```bash
MUJOCO_GL=egl uv run accelerate launch scripts/run_workspace.py \
  --config-name=train_oatpolicy task/policy=robocasa/<task>
```
Укажи токенизатор из Фазы 4. Чекпоинты по top-k SR. **Итог = `policy_robocasa_<task>.ckpt`** (запиши SR)
— база для BoN/AWR.

---

## Фаза 6 — Baseline + ГЕЙТ headroom (обязательно)

```bash
MUJOCO_GL=egl uv run scripts/eval_policy_sim.py \
  -c my_models/policy_robocasa_<task>.ckpt -o eval_out/rc_base -n 3 \
  --entropy_threshold 0 --use_k_tokens 8
```
- Полный бюджет, один сэмпл, без адаптивности = чистая база `SR_base`.
- **ГЕЙТ:** если `SR_base` очень низкий (напр. <0.2) → мало headroom для BoN → **смени задачу на
  более простую (Р1)** или больше демо. Не иди в BoN/AWR на «полу».

---

## Фаза 7 — BoN (без обучения, только eval)

```bash
for N in 4 8 16; do
  MUJOCO_GL=egl uv run scripts/eval_policy_sim.py \
    -c my_models/policy_robocasa_<task>.ckpt -o eval_out/rc_bon$N -n 3 \
    --bon_free $N --bon_signal vote --use_k_tokens 8
done
```
Ожидание (как на LIBERO): монотонный рост с насыщением, `SR(N)≈SR_base+a·log N`, +0.08..0.13, плато ~N16.
**Главный вывод:** `SR(BoN) > SR_base` → отбор превышает базу → эффект переносится.

| N | 1 (база) | 4 | 8 | 16 |
|---|---|---|---|---|
| SR | `SR_base` | ? | ? | ? |

> ⚠️ **НЕ `--bon_cap`** (даёт нули на длинных эпизодах).

---

## Фаза 8 — AWR (сбор → обучение → замер)

```bash
# 8.1 собрать (роллауты С BoN-отбором, метка = успех эпизода). Требует 2F.
MUJOCO_GL=egl uv run python scripts/collect_awr_dataset.py \
  -c my_models/policy_robocasa_<task>.ckpt -o my_datasets/rc_awr_bon.npz \
  --n_chunks 20000 --n_tasks 1 --bon_n 8 --n_workers 6
uv run python scripts/validate_awr.py -i my_datasets/rc_awr_bon.npz   # покрытие/NaN/per-episode SR

# 8.2 обучить AWR (взвешенный SFT AR-головы; зрение+токенизатор заморожены)
uv run python scripts/train_awr.py -i my_datasets/rc_awr_bon.npz \
  -c my_models/policy_robocasa_<task>.ckpt -o my_models/policy_rc_awr.ckpt \
  --beta 0.5 --beta_kl 0.05 --epochs 100 --ordering uniform

# 8.3 замерить (один сэмпл, как база — apples-to-apples)
MUJOCO_GL=egl uv run scripts/eval_policy_sim.py \
  -c my_models/policy_rc_awr.ckpt -o eval_out/rc_awr -n 3 \
  --entropy_threshold 0 --use_k_tokens 8
```
Ожидание: AWR single-sample **выше базы**, близко к inference-BoN N8, на ~0.03–0.05 ниже (запекли
отбор в один проход без удорожания вывода).

| режим | SR | стоимость вывода |
|---|---|---|
| база (1 сэмпл) | `SR_base` | 1× |
| AWR (1 проход) | ? | 1× |
| inference-BoN N=8 | ? | 8× ген. |

---

## Что прислать по итогу
1. `SR_base` (Фаза 6). 2. BoN-таблица N=1/4/8/16 + подтверждение `SR(BoN)>SR_base`.
3. AWR single-sample vs база vs BoN-N8. 4. Логи `train_awr` (baseline/weight) + `validate_awr`.
5. Расхождения с LIBERO-ожиданиями — с сырыми числами.

**Для статьи:** BoN>база И AWR дистиллирует в один проход на ВТОРОЙ среде (RoboCasa) → эффект =
**свойство класса closed-loop chunk-политик**, не одной LIBERO-политики (главный аргумент широты).

---

## Грабли (RoboCasa-специфика)
- **Экшн-DoF (Р3)** — главный риск: убедись, что fixed-base OSC_POSE даёт ровно 7D. base/torso-DoF → стоп.
- **Версии robosuite/mujoco** — RoboCasa требует свой форк robosuite; изолируй окружение от OAT/LIBERO.
- **Тяжёлая рандомизация сцен** — RoboCasa рандомит кухни/текстуры → нужен БОЛЬШОЙ датасет (mg ~3000/task),
  иначе политика недоучится → низкий SR → нет headroom.
- **Всегда `MUJOCO_GL=egl`**; **не `--bon_cap`**; `-n`=число экспериментов (не эпизодов).
- **Apples-to-apples**: база и AWR — одинаково (`--entropy_threshold 0 --use_k_tokens 8`); BoN — те же
  `--use_k_tokens 8` + `--bon_free N`.
- **pow2 токенизатор** (`num_registers: 8`) — держи, иначе бюджеты k разъедутся с LIBERO-анализом.
- Начни с **ОДНОЙ простой задачи**, доведи весь пайплайн до конца, только потом расширяй.

> ⚠️ Установочные/датасетные команды RoboCasa — по общим знаниям бенчмарка; сверь с README
> установленной версии RoboCasa/robosuite перед запуском.
```
