# Resolution plan — ICRA 2027 (RoboMimic / MetaWorld track)

**Paper rule:** в статью только `output/eval/matched_s10000/...`  
(сид отчёта `10000–10049`).

Старое `output/eval/matched/` (seed 1000) = лабораторный черновик, **не paper**.

## Протокол paper

| Param | Value |
|-------|--------|
| `test_start_seed` | **10000** |
| `n_test` | 50 |
| `-n` | **5** |
| OAT8 | `--use_k_tokens 8 --entropy_threshold 0` |
| sampling | `--temperature 1.0 --topk 10` |
| BoN | `--bon_free 8 --bon_signal vote` |
| Δ | method − matched baseline на **том же** seed pool |
| OUT | `output/eval/matched_s10000/<suite>/` |

## Seed pools — anti-leak (зафиксировано 2026-07-16, уточнено RM vs MW)

Оба раннера (`RoboMimicRunner` / `MetaworldRunner`) default **`test_start_seed=1000`**.  
В сохранённых paper-ckpt **нет** override `test_start_seed` — TopK на фите у всех с 1000.  
Разные фиты (RM `cfg.seed=42`, MW `cfg.seed=0`) = torch/dataloader RNG, **не** env-episode seeds.

| Пул | Seeds | Роль | Eval (report) | AWR collect |
|-----|-------|------|---------------|-------------|
| **selection RM** | `1000–1049` (`n_test=50`) | TopK train-time | ❌ | ❌ |
| **selection MW** | `1000–1249` (`n_test=250`) | TopK train-time | ❌ | ❌ |
| **paper report** | `10000–10049` | Table P / matched Δ | ✅ только сюда | ❌ |
| **collect** | `0` (+ worker offsets) | offline AWR data | — | ✅ Wave 2 |

Правила:
1. **Все** paper eval (baseline / BoN / AWR) → `test_start_seed=10000` (вне RM и MW selection).
2. AWR collect → `--seed 0`; **не** `1000…1249`, **не** `10000`.
3. Paper-артефакты Wave1 = `baseline_n5/eval_log.json` + `bon_n8_n5/eval_log.json` + `summary.json` (+ Wave1 log с `DONE bon`). Отсутствие строки `ALL DONE` в логе (bash mid-edit) **не** инвалидирует eval_logs.

В тексте: *selected on train-time pool starting at 1000 (RM …1049 / MW …1249), reported on 10000–10049*.

## Что ещё открыто (не seed-leak)

| Item | Статус | В статье |
|------|--------|----------|
| **MetaWorld demo port** (regen Zarr vs paper sim/success) | контролируемый limitation | да, Limitations: MW interpret within our port |
| **Lift** | нужен retrain → потом Wave 1–2 @ s10000 | TBD |
| **Square** | Wave 1 @ ep-1500; если BoN flat → rematch ep-0600 @ s10000 | TBD |
| **MT4 multitask** | exploratory only | **не в paper** |

Can + MW specialists (coffee/stick/disassemble/box) при закрытом Wave 1 — seed-чистые для Table P.

## Запуск

```bash
# с хоста — все NOW suite параллельно (baseline+BoN, без старого AWR):
bash oat/scripts/cluster_launch_matched_paper_wave.sh
```

## ⛔ GATE перед Wave 2 (AWR) — зафиксировано 2026-07-16

**Не стартовать AWR, пока Wave 1 paper не закрыт.** В Wave 2 **запрещены** ранние/exploratory раны.

### Обязательно до Wave 2

1. У suite есть `output/eval/matched_s10000/<suite>/summary.json` с **baseline_n5 + bon_n8_n5** (оба `eval_log.json`).
2. Числа вписаны в **Table P** (`RESULTS.md`) — baseline, BoN, Δ_BoN. Источник **только** `matched_s10000`, не seed-1000, не chain5, не exploratory.
3. Решение по suite: BoN дал смысл → идём в AWR; flat/анти → AWR пропускаем или (Square) fallback ckpt, см. ниже.
4. Старые артефакты **не** переиспользовать:
   - ❌ `my_models/policy_awr_*.ckpt` (exploratory)
   - ❌ `my_datasets/awr_*.npz` со старых пайплайнов
   - ❌ `cluster_*_bon_awr*.sh` как paper-path (у них eval/seed не paper)
   - ❌ Table B / exploratory SR в текст или как baseline для Δ_AWR

### Wave 2 правила (свежий AWR only)

| Этап | Правило |
|------|---------|
| Collect | `--seed 0` явно (лог в RESULTS). ❌ `1000` (selection), ❌ `10000` (report) |
| Train | новый `my_models/awr_s10000_<suite>.ckpt` |
| Eval | только через triplet: `AWR_CKPT=... SKIP_AWR=0 TEST_START_SEED=10000` → `awr_n5/` |
| Δ_AWR | AWR − **тот же** paper baseline из Wave 1 |

```bash
# после GATE — один suite, anti-leak names + seed 0 collect + eval @10000:
SUITE=box-close GPU=0 bash scripts/cluster_matched_paper_wave2_awr.sh
# пишет: my_datasets/awr_s10000_<suite>.npz
#         my_models/awr_s10000_<suite>.ckpt
#         matched_s10000/<suite>/awr_n5/  (SKIP_BASELINE_BON=1 — не трёт Wave1)
```

### Square contingency

- Paper first-try: **ep-1500**. Если BoN@s10000 flat → rematch **ep-0600** тем же paper-протоколом; в Table P — один победивший ckpt.
- Exploratory BoN/AWR @ ep-0600 **не** подставлять в Table P.

### OAT Table VI (оригинал) = sanity only

Lift 99.2 / Can 80.8 / Square 39.2; MW box 44.4 / coffee 26.4 / disassemble 17.2 / stick 9.6.  
**Не** comparator для Δ; absolute parity не claim.

## ⛔ Latency / Table C — протокол репрезентативности (зафиксировано 2026-07-17)

Latency **нужна для статьи** (BoN = N× AR на replan; AWR = single-forward).  
Это **отдельный** policy-forward замер — **не** перепрогон matched SR. Table P SR остаётся из `matched_s10000/.../eval_log.json`.

**Когда:** после Wave 1–2 SR для suite’ов, которые идут в paper (Phase 5).  
**Что мерим:** Single (baseline OAT8) / BoN N=8 vote / AWR — wall-clock **ms per policy call**, batch=1.  
**Куда:** `output/eval/matched_s10000/<suite>/latency.json` → Table C (SR из Table P × ms × ΔSR × cost).  
**Скрипт:** `scripts/measure_latency_paper.py` + `scripts/cluster_latency_paper_done.sh` (тот же docker/кластер, что paper eval). Docker без `.git` → launch с `OAT_GIT_COMMIT`/`OAT_GIT_BRANCH`/`OAT_GIT_DIRTY` с host. Paper-proof refuse если `git_commit` пустой.

### Обязательные условия (иначе не в статью)

1. **Тот же hardware / окружение**, на котором интерпретируем SR  
   Один GPU (тот же тип, что paper eval: cluster V100 / тот же docker image), тот же CUDA/driver stack.  
   ❌ Не мерить latency на Mac/другой машине и стыковать с cluster SR.  
   ❌ Не менять TensorRT/ONNX/compile path относительно того, как реально крутится `eval_policy_sim` (сейчас = обычный PyTorch forward; если позже появится TRT — мерить **и** SR-путь, **и** latency на нём же).

2. **Batch=1 + максимально deterministic**  
   `batch_size=1`. Где возможно: `torch.backends.cudnn.deterministic` / fixed seeds для timing-loop (не путать с env `test_start_seed` Table P).  
   Warmup forward’ы **исключить** из статистики (отдельный warmup, потом timed reps).

3. **Несколько повторов → median и mean±std**  
   ≥ **5–10** timed runs на режим (Single / BoN / AWR). В Table C: **median** как основная цифра + **mean±std** (или IQR) в скобках/appendix.  
   ❌ Один «случайный» прогон в paper не годится.

4. **Тот же input pipeline, что sim-eval**  
   Та же предобработка obs (resize, нормализация, To-stack, dtype/device), что `obs_encoder` видит в `eval_policy_sim` / runner.  
   Obs брать из checkpoint dataset / реальных val-батчей (как `--obs_from_checkpoint_dataset` в adaptive latency), не «пустой» dummy random, если он меняет путь/entropy.  
   Режим генерации: Single = тот же OAT8 (`use_k_tokens=8`, `entropy_threshold=0`, T=1, topk=10); BoN = `bon_free=8`, `bon_signal=vote`; AWR = single-sample на `awr_s10000_<suite>.ckpt`.

5. **AWR ckpt = тот же Wave 2 артефакт, та же архитектура**  
   Путь AWR для latency = ровно `my_models/awr_s10000_<suite>.ckpt`, который дал `matched_s10000/<suite>/awr_n5/` (сверить path + mtime/sha с Wave2 log / `summary.json`).  
   Baseline/BoN latency — с того же `BASE_CKPT`, что Wave1.  
   ❌ Не подставлять exploratory `policy_awr_*`, другой epoch, или ckpt после рефактора policy/encoder/tokenizer — иначе ms не из того же семейства, что Table P SR.  
   Если код модели менялся после Wave2 eval — либо откат к commit Wave2, либо **пересчёт SR** на новом коде (не смешивать).

6. **Воспроизводимость: версия кода в `latency.json`**  
   В каждый `matched_s10000/<suite>/latency.json` писать как минимум:  
   `git_commit` (полный hash), `git_dirty` (bool), `git_branch`, `measured_at` (ISO), `host` / `gpu_name`, `torch_version` / `cuda_version`, пути `base_ckpt` / `awr_ckpt` (+ optional sha256), режимы и N reps / median / mean / std, `paper_proof: true`, `fairness.obs_counter_reset_per_mode`.  
   Цель — точно воспроизвести замер; без commit latency не считать paper-final.  
   **Fairness:** перед каждым режимом (Single / BoN / AWR) сбрасывать obs-counter и делать свой warmup — иначе BoN «наследует» прогретый Single.

### Явно не делать

- ❌ Не пересчитывать Table P SR ради latency.  
- ❌ Не подмешивать sim wall-clock (MuJoCo/render) в «policy latency» — Table C = **inference / policy-forward cost**, с оговоркой что episode wall-clock доминирует sim.  
- ❌ Не сравнивать latency между suite’ами как абсолютный SOTA без одной машины/одного протокола.  
- ❌ Не мерить AWR latency на другом ckpt/архитектуре, чем Wave2 Table P.

В тексте статьи: *latency measured on the same cluster GPU/stack as matched eval; batch=1; median over N timed forwards; same obs pipeline as sim-eval; same Wave2 AWR ckpt; code commit recorded in latency.json; SR from Table P unchanged.*

## Suite статус

| Suite | Paper now |
|-------|-----------|
| Can, coffee, stick, disassemble, box-close | Wave1 BoN DONE → **Wave2 AWR running** (`paper_w2_*`) |
| Square | Wave1 BoN @ ep-1500 running |
| Lift | после retrain |
| MT4 | не в paper |

См. Table P в [`RESULTS.md`](RESULTS.md).
