---
license: mit
task_categories:
  - robotics
language:
  - en
tags:
  - oat
  - imitation-learning
  - robomimic
  - metaworld
  - robocasa
  - zarr
  - hdf5
pretty_name: AAAI OAT Paper Training Datasets (HDF5 + Zarr)
size_categories:
  - 10G<n<100G
---

# AAAI paper datasets — OAT training data (HDF5 sources + Zarr)

Canonical **training datasets** used to reproduce our AAAI submission  
*"Spend Compute on Selection, Not Reduction"* (Consensus Selection / CS-D on **OAT** policies).

**Hub repo (datasets):** [`hackhackhack66666/aaai-datasets`](https://huggingface.co/datasets/hackhackhack66666/aaai-datasets)  
**Hub repo (models + eval artifacts):** [`hackhackhack66666/aaai27-models`](https://huggingface.co/hackhackhack66666/aaai27-models) — **primary copy** for checkpoints, Table P eval, latency, logs.  
**This dataset repo contains only raw + converted training data** (HDF5 where retained, Zarr for all suites).

---

## Quick map — what is used in the paper?

| Suite | Zarr (train) | HDF5 (source) | In paper **Table P**? | Paper role |
|-------|--------------|---------------|----------------------|------------|
| **RoboMimic** Lift | `robomimic/zarr/lift_N200.zarr` | `robomimic/hdf5/lift_mh_image.hdf5` | **Yes** (Lift run B ep-1400) | Generalization baseline |
| **RoboMimic** Can | `robomimic/zarr/can_N200.zarr` | `robomimic/hdf5/can_mh_image.hdf5` (+ raw `can/mh/*`) | **Yes** | Generalization baseline |
| **RoboMimic** Square | `robomimic/zarr/square_N200.zarr` | `robomimic/hdf5/square_mh_image.hdf5` | **Yes** | Generalization baseline |
| **MetaWorld** coffee-pull | `metaworld/zarr/coffee-pull_N50.zarr` | — (generated) | **Yes** | Generalization baseline |
| **MetaWorld** stick-pull | `metaworld/zarr/stick-pull_N50.zarr` | — | **Yes** | Generalization baseline |
| **MetaWorld** disassemble | `metaworld/zarr/disassemble_N50.zarr` | — | **Yes** | Generalization baseline |
| **MetaWorld** box-close | `metaworld/zarr/box-close_N50.zarr` | — | **Yes** | Generalization baseline |
| **MetaWorld MT4** multitask | `metaworld/zarr/mt4_N50.zarr` | — | **No** (exploratory only) | Early multitask probe; **not** Table P |
| **RoboCasa** close_drawer | `robocasa/zarr/close_drawer_N200.zarr` | partial human HDF5 | **Yes** | Generalization baseline |
| **RoboCasa** coffee_press_button | `robocasa/zarr/coffee_press_button_N200.zarr` | partial human HDF5 | **Yes** | Generalization baseline |
| **RoboCasa** turn_off_sink_faucet | `robocasa/zarr/turn_off_sink_faucet_N200.zarr` | — (zarr only) | **Yes** | Generalization baseline |
| **RoboCasa** turn_off_microwave | `robocasa/zarr/turn_off_microwave_N200.zarr` | — (zarr only) | **Yes** | Generalization baseline |
| **LIBERO** libero10 legacy | `libero/zarr/libero10_N500.zarr` | — | **No** | Legacy OAT smoke / auxiliary dataset |

**LIBERO note:** this repo also stores one **legacy OAT-format** LIBERO zarr (`libero10_N500.zarr`) as a backup / smoke dataset. It is **not** the paper Table P data and **not** the `physical-intelligence/libero` parquet format.

---

## Directory layout

```text
robomimic/
  zarr/{lift,can,square}_N200.zarr/     # OAT training format (200 demos each)
  hdf5/
    lift_mh_image.hdf5
    can_mh_image.hdf5
    square_mh_image.hdf5
    can/mh/demo_v15.hdf5                  # raw multi-human before image extract
    can/mh/image_v15.hdf5

metaworld/
  zarr/
    mt4_N50.zarr                          # 200 eps = 50×4 tasks (multitask)
    {box-close,coffee-pull,disassemble,stick-pull}_N50.zarr
  METAWORLD_GENERATION.md                 # how demos differ from upstream sim-env

robocasa/
  zarr/<task>_N200.zarr/                  # each includes ROBOCASA_SOURCE.txt
  hdf5/
    CloseDrawer/human/demo_gentex_im128_randcams.hdf5
    CoffeePressButton/human/demo_gentex_im128_randcams.hdf5

libero/
  zarr/libero10_N500.zarr/                # legacy OAT-format backup (500 eps)
```

Approximate sizes (cluster, 2026-08-13):

| Path | ~Size |
|------|-------|
| RoboMimic HDF5 total | 10.5 GB |
| RoboMimic Zarr total | 1.0 GB |
| MetaWorld Zarr total | 1.4 GB |
| RoboCasa Zarr total | 4.6 GB |
| RoboCasa HDF5 (retained) | 0.6 GB |
| LIBERO legacy Zarr | 3.4 GB |
| **Total** | **~22 GB** |

---

## Common Zarr schema (OAT)

All Zarr stores follow the OAT pipeline layout:

- `data/` — time-major arrays (`action`, RGB, proprioception)
- `meta/episode_ends` — cumulative step indices per episode
- RoboCasa only: `ROBOCASA_SOURCE.txt` inside each zarr (provenance + sha256 in paper log)

Validate locally (from `oat/` repo):

```bash
python scripts/validate_robomimic_data.py
python scripts/validate_robocasa_data.py
python scripts/validate_metaworld_data.py data/metaworld/box-close_N50.zarr --num-tasks 1
python scripts/validate_metaworld_data.py data/metaworld/mt4_N50.zarr --num-tasks 4 --require-subtask-counts
```

---

## 1. RoboMimic (official mh demos → Zarr)

### Source

- Benchmark: **RoboMimic multi-human (mh)** image demonstrations.
- Tasks: **Lift**, **Can**, **Square**.
- **200 demonstrations** per task (subsampled from 300 mh demos).
- Lift: direct download `lift_mh_image.hdf5`.
- Can / Square: download raw `demo_v15.hdf5`, replay through robosuite 1.5 to extract `image_v15.hdf5` (see `scripts/extract_robomimic_mh_image.sh`).

### Zarr keys (train)

| Key | Shape (per step) |
|-----|------------------|
| `action` | 7 |
| `agentview_image` | 84×84×3 |
| `robot0_eye_in_hand_image` | 84×84×3 |
| (+ proprio keys per converter) | |

### Paper linkage

- Tokenizer: top MSE checkpoint per task (`ep-0690` square, `ep-1970` lift, etc.).
- Policy TopK @ `test_start_seed=1000`; **Table P report** @ `test_start_seed=10000`, `-n 5`, OAT8.
- Locked Table P ckpts: Can ep-1700, Square ep-0700, Lift ep-1400 (run B).

### Notes

- **Do not** use legacy `OAT-RoboMimic-Fine-tune/BLT-OAT/data/robomimic/` (old ph layout) — unrelated to this paper track.

---

## 2. MetaWorld (locally generated expert demos → Zarr)

### ⚠️ Important — not the official MetaWorld HDF5 download

All MetaWorld data here was **generated on our cluster** with:

```bash
python scripts/gen_metaworld_data.py --task_name <task> --num_episodes 50 --force
```

(log: `logs/metaworld_single_data_regen.log`; multitask: `logs/gen_metaworld_mt4_N50.log`)

Environment code is ported from **[sim-env](https://github.com/Chaoqi-LIU/sim-env)** into `oat/oat/env/metaworld/` (MuJoCo **2.1.0** / MetaWorld v2 stack — **different** from RoboMimic/LIBERO robosuite 1.4).

### Demo acceptance rule (differs from stricter filters)

An episode is kept iff **`info["success"] == True` on at least one timestep** during the expert rollout:

```python
episode_success_count += int(bool(info.get("success", False)))
if episode_success_count == 0:
    continue  # reject episode, retry with new seed
```

We **do not** require success only on the terminal step. This matches **`mt4_N50`** generation and the four **single-task** regen runs used for Table P.

### Reset / seed fix (July 2026)

Single-task regen originally hung because `MetaworldEnv.reset()` without `seed` restored a **fixed** MuJoCo snapshot. Fix:

```python
roll_seed = episode_idx * 1_000_000 + attempt_idx
obs_dict, _ = env.reset(seed=roll_seed)
```

Without incrementing `attempt_idx` on reject, retries repeat the same failed init.

### Tasks & episode counts (validated 2026-08-13)

| Zarr | Episodes | Action dim | Cameras | Steps (total) |
|------|----------|------------|---------|---------------|
| `mt4_N50.zarr` | 200 (50×4) | 4 | 4×128² RGB + `agent_pos` 9D | 20 316 |
| `box-close_N50.zarr` | 50 | 4 | same | 5 616 |
| `coffee-pull_N50.zarr` | 50 | 4 | same | 4 088 |
| `disassemble_N50.zarr` | 50 | 4 | same | 5 220 |
| `stick-pull_N50.zarr` | 50 | 4 | same | 6 101 |

MT4 task order in `subtask_counts`: `[box-close, coffee-pull, disassemble, stick-pull]`.

Single-task files can also be obtained deterministically by splitting `mt4_N50.zarr` (`scripts/split_metaworld_mt4_zarr.py`, round-robin `i % 4`); **paper Table P uses freshly regen single-task zarr**, byte-identical to split for the episodes that were completed before regen.

### How this differs from “original sim-env / paper MetaWorld”

1. **Success timing:** our collector accepts **any-time success** during the demo rollout (see above). A stricter **terminal-only** filter was tested and rejected for compatibility with `mt4_N50`.
2. **Implementation port:** same *intent* as sim-env, but our wrapper had the **reset-without-seed bug** (fixed before final single-task zarr). Treat MW numbers as **valid within this port**, not as a claim of bit-identical reproduction of Chaoqi sim-env demo files.
3. **Paper limitation (explicit):** *“MetaWorld demo port — controlled limitation; interpret MW within our implementation.”* (see `RESULTS.md` / `RESOLUTIONPLAN.md` in code repo).

### Paper linkage

- **Table P** uses **single-task specialists** (one zarr → one tokenizer → one policy per task).
- `mt4_N50.zarr` is **exploratory multitask** (shared model); **not** reported in Table P.

---

## 3. RoboCasa (official v0.2 HDF5 → Zarr)

### Source (G0 protocol)

- RoboCasa **v0.2** registry: `human_im` + `mg_im` from UT Austin Box (`demo_gentex_im128_randcams.hdf5`).
- Mix per task: **50 human + 150 MimicGen**, subsample **seed 0**, **action_dim = 12**.
- Converter: `scripts/convert_robocasa_dataset.py`.

### Tasks (paper Table P)

| Task slug | Zarr | Human+MG eps | TopK lock (selection @ seed 2000) |
|-----------|------|--------------|-----------------------------------|
| `close_drawer` | `close_drawer_N200.zarr` | 200 | ep-0500 @ SR 0.700 |
| `coffee_press_button` | `coffee_press_button_N200.zarr` | 200 | ep-0500 @ SR 0.600 |
| `turn_off_sink_faucet` | `turn_off_sink_faucet_N200.zarr` | 200 | ep-0500 @ SR 0.580 |
| `turn_off_microwave` | `turn_off_microwave_N200.zarr` | 200 | ep-0500 @ SR 0.620 |

Each zarr contains `ROBOCASA_SOURCE.txt` listing exact human/MG HDF5 keys used before MG files were deleted on cluster (disk policy).

### HDF5 retention policy on cluster

After Zarr conversion we **deleted MimicGen HDF5** to save disk. **Human** HDF5 for **CloseDrawer** and **CoffeePressButton** were kept (~614 MB).  
**Microwave** and **sink** human HDF5 were also removed post-convert — **full training data is in Zarr only** for those two tasks.

### Obs keys (Zarr)

`action`, `robot0_agentview_{left,right}_rgb`, `robot0_eye_in_hand_rgb`, `robot0_eef_pos`, `robot0_eef_quat`, `robot0_gripper_qpos` (128×128 RGB).

### Paper eval protocol (not in this repo)

- Selection TopK: `test_start_seed=2000`, `n_test=50`.
- Table P report: literal seeds **`10000…10004`**, each `-n 1 --n_test 50`, OAT8 + BoN8 `vote`.

---

## 4. LIBERO legacy backup (OAT zarr)

This repository also stores a **legacy OAT-format LIBERO dataset**:

- `libero/zarr/libero10_N500.zarr`
- size: ~3.4 GB
- episodes: **500**
- total steps: ~138k
- action dim: **7**

Observed keys:

- `action`
- `agentview_rgb`
- `prompt`
- `robot0_eef_pos`
- `robot0_eef_quat`
- `robot0_eye_in_hand_rgb`
- `robot0_gripper_qpos`
- `robot0_joint_pos`
- `task_uid`

Important:

1. This is a **converted OAT zarr**, not `physical-intelligence/libero` parquet.
2. It is useful for **fast sanity checks / smoke runs** only.
3. Final conclusions for new LIBERO projects should still use the intended upstream format/revision.

---

## Regeneration pointers (code repo)

| Suite | Script |
|-------|--------|
| MetaWorld gen | `scripts/gen_metaworld_data.py`, `scripts/cluster_gen_metaworld_single_data.sh` |
| MetaWorld split | `scripts/split_metaworld_mt4_zarr.py` |
| RoboMimic convert | `scripts/convert_robomimic_dataset.py`, `scripts/prepare_robomimic_{lift,can,square}.sh` |
| RoboCasa convert | `scripts/convert_robocasa_dataset.py` |

Cluster path when uploaded (2026-08): `/home/askhabaliev_gs/mipt_paper/oat/data/`.

---

## Citation

If you use these datasets, cite **OAT** (ordered action tokenization) and our AAAI paper (anonymous submission at upload time). RoboMimic / MetaWorld / RoboCasa have their own benchmark citations — see respective papers.

---

## Changelog

| Date | Note |
|------|------|
| 2026-08-13 | Initial public upload: all paper Zarr + retained HDF5 from MIPT cluster `mipt_paper/oat`. |
| 2026-08-19 | Added legacy `libero10_N500.zarr` backup from `~/OAT/data/libero/` for smoke / recovery use. |
