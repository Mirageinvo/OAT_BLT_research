# Hackathon — RoboMimic baseline (paper HF ckpts)

**Goal:** работать с **опубликованными baseline-моделями** с Hugging Face — воспроизвести eval, показать sim-видео и слайды по обучению.

## Hackathon POC (2 tasks, 1 GPU, no square)

Нет времени на square — только **lift + can**, одна GPU, **2.5h total** (~1.25h на задачу):

```bash
cd oat
export MUJOCO_GL=egl
bash scripts/hackathon/00_download_assets.sh
# HDF5: data/robomimic/hdf5_datasets/{lift,can}_mh_image.hdf5

bash scripts/hackathon/run_hackathon_poc.sh
```

- Train sequential на GPU 0 (без OOM)
- Quick eval `n_test=10` → грубый SR в `eval_log.json`
- 1 success gif на задачу (80 шагов после success)

```bash
TOTAL_SEC=9000 GPU=0 N_TEST=10 bash scripts/hackathon/run_hackathon_poc.sh
```

## Timed parallel train (3 GPU, all 3 tasks)

```bash
bash scripts/hackathon/00_download_assets.sh
bash scripts/hackathon/run_parallel_timed.sh   # TIME_LIMIT_SEC=9000, TRAIN_GPUS=0,1,2
```

## Три задачи (canonical)

| Task | Baseline SR | Tokenizer | Policy | Dataset |
|------|-------------|-----------|--------|---------|
| **Lift** — поднять куб | **82.0 ± 3.2%** | `tokenizers/lift/ep-1970_mse-0.006.ckpt` | `policies/lift/ep-0900_sr-0.930.ckpt` | `lift_N200.zarr` |
| **Can** — поднять банку | **76.4 ± 3.0%** | `tokenizers/can/ep-0520_mse-0.005.ckpt` | `policies/can/ep-1700_sr-0.940.ckpt` | `can_N200.zarr` |
| **Square** — pick+place | **36.0 ± 7.5%** | `tokenizers/square/ep-0690_mse-0.004.ckpt` | `policies/square/ep-0700_sr-0.420.ckpt` | `square_N200.zarr` |

## Hugging Face

| Что | Repo |
|-----|------|
| Tokenizers + policies | [`hackhackhack66666/robomimic-oattok-policy`](https://huggingface.co/hackhackhack66666/robomimic-oattok-policy) |
| Zarr (200 demos) | [`hackhackhack66666/robomimic_zarr`](https://huggingface.co/datasets/hackhackhack66666/robomimic_zarr) |
| Eval + sim-видео | [`hackhackhack66666/aaai27-models`](https://huggingface.co/hackhackhack66666/aaai27-models) |

Eval paths на HF:
- `eval/matched_s10000/{lift,can,square}/summary.json`
- `eval/matched_s10000/{task}/baseline_n5/media/{task}/*.mp4`

Машиночитаемый манифест: `scripts/hackathon/baseline_manifest.json`

## Pipeline (GPU для eval/rollout)

```bash
cd oat
export MUJOCO_GL=egl

# 1) скачать tokenizers, policies, zarr, summary, demo mp4
bash scripts/hackathon/00_download_assets.sh

# 2) проверить + слайды «что обучали» (epoch budget, ckpt, paper SR)
bash scripts/hackathon/03_make_slides.sh

# 3) optional: воспроизвести baseline SR (protocol matched_s10000)
bash scripts/hackathon/01_eval_baseline.sh

# 4) gif для презы (из HF mp4 или fresh rollout)
bash scripts/hackathon/02_prepare_demo.sh          # MODE=hf (default)
bash scripts/hackathon/02_prepare_demo.sh MODE=rollout   # свой rollout
```

## Локальные пути после download

```
hackathon_assets/
  tokenizers/{lift,can,square}.ckpt
  policies/{lift,can,square}.ckpt      # paper baseline

hackathon_output/
  eval/matched_s10000/*/summary.json   # paper numbers (with HF)
  demo/media/{task}/*.mp4              # sim rollouts with HF
  gifs/{task}_baseline_*.gif           # для слайда Demo
  progress/
    baseline_training_cards.png        # слайд Training (ckpt, epochs, SR)
    baseline_all_summary.json
    {task}_baseline_summary.json
```

## Протокол eval (paper)

- `test_start_seed=10000`, `n_test=50`, `-n 5` (5 exp)
- OAT8: `--use_k_tokens 8 --entropy_threshold 0`
- Сравнивать с `summary.json` на HF

## Слайды презы

**Training** — `progress/baseline_training_cards.png`:
- frozen tokenizer + policy ckpt (ep 900 / 1700 / 700)
- train budget ~5001 epochs (из hydra config на HF)
- paper baseline SR (matched_s10000)

> Полные `logs.json` (loss curve) на HF **нет**. Если нужен график loss↓ — optional retrain (`01_train_policies.sh`) или локальный cluster output.

**Demo** — `gifs/*_baseline_*.gif` или success rollout

## HDF5 для sim

Нужны для eval/rollout:
```
data/robomimic/hdf5_datasets/lift_mh_image.hdf5
data/robomimic/hdf5_datasets/can_mh_image.hdf5
data/robomimic/hdf5_datasets/square_mh_image.hdf5
```

Скачать из [`aaai-datasets`](https://huggingface.co/datasets/hackhackhack66666/aaai-datasets):
```bash
hf download hackhackhack66666/aaai-datasets robomimic/hdf5/lift_mh_image.hdf5 \
  --repo-type dataset --local-dir data/robomimic/hdf5_datasets
# repeat for can, square
```

## Optional: train from scratch (hackathon variant)

Если нужен **новый** policy ckpt + loss curves — **не** paper baseline:

```bash
bash scripts/hackathon/01_train_policies.sh   # frozen tok only
bash scripts/hackathon/03_plot_progress.sh    # logs.json curves
```

Paper numbers — **только** из HF ckpts выше.

## Reviewer: Panda → UR5e (отдельно)

Lift gif с UR5e требует retrain на новом роботе; Panda weights не переносятся.
