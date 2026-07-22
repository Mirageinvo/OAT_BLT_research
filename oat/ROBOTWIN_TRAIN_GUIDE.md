# OAT on RoboTwin — download → convert → train → validate (minimal-diff package)

Goal: train + validate an OAT policy on **RoboTwin 2.0** (bimanual, SAPIEN) on a **separate
machine**, reusing the existing OAT training/eval entry points with the smallest possible diff,
then run our BoN/AWR study. RoboTwin = SAPIEN physics, dual-arm, 50 tasks, 100K demos, HDF5
(native) or LeRobot v3.0 (HF Hub). Repo: https://github.com/robotwin-Platform/robotwin ,
docs: https://robotwin-platform.github.io/doc/ .

> **What is minimal-diff vs new:**
> - **Train / validate = REUSED AS-IS** (`scripts/run_workspace.py`, `scripts/eval_policy_sim.py`) —
>   zero code change, only new task configs + a RoboTwin runner (Hydra-instantiated).
> - **Model = config only** — OAT is action-dim-agnostic below the tokenizer (AR head is
>   token-space); the only real change is `action_dim: 14` (bimanual) in the configs + a
>   tokenizer retrain. See §0.
> - **NEW code (unavoidable, SAPIEN ≠ robosuite):** `RoboTwinEnv`, `RoboTwinRunner`, dataset
>   converter. Provided as SCAFFOLDS that mirror the LIBERO files 1:1 with RoboTwin-specific
>   `# TODO` where the SAPIEN/RoboTwin API goes (fill on the RoboTwin machine).

================================================================================
## §0. ARCHITECTURE DECISIONS (make these first — they drive everything)
================================================================================
**D1 — Action dim = 14 (bimanual EEF).** RoboTwin dual-arm EEF action = 2 arms × (6D pose delta +
1 gripper) = **14**. VERIFY against RoboTwin's actual action layout (could be joint-space, e.g.
2×7 joints + 2 grippers = 16, or include a mobile base). **Pick ONE fixed `action_dim` and set it
in both configs.** OAT below the tokenizer does NOT care about the value (AR head predicts token
indices), so 14 vs 7 is a **config change**, not an architecture change.

**D2 — Tokenizer capacity.** 8 tokens now compress a 32×14 = 448-dim chunk (vs 32×7 = 224 for
LIBERO). Retrain the tokenizer (action-only, cheap) and **check reconstruction MSE**. If it is
much worse than LIBERO's ~0.002:
- bump `num_registers: 8 → 16` (and/or a larger FSQ codebook) in `train_oattok.yaml` override.
- This propagates to the policy's `max_seq_len` (8 → 16) — **config only**, AR head just predicts
  16 tokens. Keep `token_dropout_mode: 'pow2'` for consistency with our K-analysis (our K-axis
  study is on LIBERO; RoboTwin only needs a working policy for the BoN/AWR breadth result).

**D3 — Cameras = 2.** RoboTwin is multi-view; OAT's obs encoder expects **2 RGB cameras**. Pick 2
(e.g. a global/front + one wrist) and expose them under the OAT keys `agentview_rgb` +
`robot0_eye_in_hand_rgb` in `RoboTwinEnv._extract_obs` and the converter. (Extending to 3 cameras
= changing shape_meta + encoder; avoid.)

**D4 — Proprio.** Dual-arm → concatenate both arms' eef pose/gripper (or joint states) into the
`state` ports. Set the `shape` in shape_meta to the real concatenated dim.

================================================================================
## §1. INSTALL (RoboTwin machine)
================================================================================
```bash
# 1) OAT deps (isolated env — SAPIEN/CuRobo/pytorch3d conflict with robosuite/LIBERO)
conda create -n oat_robotwin python=3.10 -y && conda activate oat_robotwin
cd oat && uv sync   # or pip install -e . ; then add RoboTwin deps below
# 2) RoboTwin + SAPIEN stack (per their README — SAPIEN, CuRobo, mplib, pytorch3d)
git clone https://github.com/robotwin-Platform/robotwin && cd robotwin
bash script/install.sh   # or follow robotwin-platform.github.io/doc/usage/... (verify script name)
```
Sanity: run a RoboTwin demo/eval from their repo to confirm SAPIEN renders before porting.

================================================================================
## §2. DOWNLOAD DATASET
================================================================================
**Option A (recommended, easiest) — pre-collected LeRobot v3.0 dataset (HF Hub, 79.6 GB):**
```bash
# via huggingface_hub / lerobot; dataset id per RoboTwin docs (verify exact repo id)
huggingface-cli download --repo-type dataset robotwin/robotwin-2.0 --local-dir data/robotwin_lerobot
```
**Option B — native HDF5 (RoboTwin data-collection pipeline, closer to our LIBERO converter):**
```bash
# generate/collect demos for one task (see robotwin-platform.github.io/doc/usage/collect-data.html)
python script/collect_data.py --task <task_name> --num_episodes 500   # verify args
# -> per-episode HDF5 files under data/<task>/ , instructions under instructions/
```
Start with **ONE task** (a simple pick/place, fixed-base bimanual) end-to-end before scaling.

================================================================================
## §3. CONVERT TO ZARR  (new file, mirrors LIBERO converter)
================================================================================
Files provided (fill the RoboTwin-specific TODOs):
- `oat/env/robotwin/dataset_conversion.py` — `convert_robotwin_to_zarr(...)`
- `scripts/convert_robotwin_dataset.py` — CLI

```bash
# HDF5 path (Option B):
MUJOCO_GL=egl uv run python scripts/convert_robotwin_dataset.py \
    --src data/<task> --task <task> -n 500 --format hdf5
# LeRobot path (Option A): --format lerobot --src data/robotwin_lerobot --task <task>
# -> data/robotwin/<task>_N<n>.zarr  (matches the zarr_path in the robotwin task configs)
```
Zarr schema MUST be: `action[14]`, `agentview_rgb[H,W,3]`, `robot0_eye_in_hand_rgb[H,W,3]`,
`robot0_eef_pos/quat/gripper_qpos` (or dual-arm equivalents), `task_uid[1]`. TODOs in the module:
action slicing to 14D (D1), camera keys (D3), proprio concat (D4), image flip/orientation.

================================================================================
## §4. TRAIN TOKENIZER  (REUSED entry point, new config)
================================================================================
```bash
MUJOCO_GL=egl uv run accelerate launch scripts/run_workspace.py \
    --config-name=train_oattok task/tokenizer=robotwin/<task>
```
- Dumps recon-MSE — **check it (D2)**. If poor, re-run with `tokenizer.encoder.num_registers=16`
  override. Freeze the tokenizer checkpoint for stage 2.

================================================================================
## §5. TRAIN POLICY  (REUSED entry point, new config)
================================================================================
```bash
MUJOCO_GL=egl uv run accelerate launch scripts/run_workspace.py \
    --config-name=train_oatpolicy task/policy=robotwin/<task>
```
- Point at the tokenizer from §4 (config or override). Checkpoints selected by top-k sim SR
  (needs `RoboTwinRunner`, §7). Output: `policy_robotwin_<task>.ckpt`.

================================================================================
## §6. VALIDATE + BoN/AWR  (REUSED scripts, zero diff)
================================================================================
```bash
# baseline (single-sample, full budget) — sanity + headroom gate
MUJOCO_GL=egl uv run scripts/eval_policy_sim.py -c my_models/policy_robotwin_<task>.ckpt \
    -o eval_out/rt_base -n 3 --entropy_threshold 0 --use_k_tokens 8
# BoN N=8 (vote)   |   AWR: collect -> train_awr -> eval  (identical to LIBERO)
MUJOCO_GL=egl uv run scripts/eval_policy_sim.py -c my_models/policy_robotwin_<task>.ckpt \
    -o eval_out/rt_bon8 -n 3 --bon_free 8 --bon_signal vote --use_k_tokens 8
MUJOCO_GL=egl uv run python scripts/collect_awr_dataset.py -c my_models/policy_robotwin_<task>.ckpt \
    -o my_datasets/rt_awr.npz --n_chunks 20000 --n_tasks 1 --bon_n 8 --n_workers 6
uv run python scripts/train_awr.py -i my_datasets/rt_awr.npz -c my_models/policy_robotwin_<task>.ckpt \
    -o my_models/policy_rt_awr.ckpt --beta 0.5 --beta_kl 0.05 --epochs 100
MUJOCO_GL=egl uv run scripts/eval_policy_sim.py -c my_models/policy_rt_awr.ckpt \
    -o eval_out/rt_awr -n 3 --entropy_threshold 0 --use_k_tokens 8
```
`eval_policy_sim.py` instantiates the runner from `cfg.task.policy.env_runner` via Hydra → picks
up `RoboTwinRunner` automatically. `--bon_free` etc. work unchanged (BoN/AWR are action-dim-
agnostic). NOTE: `collect_awr_dataset.py` imports LIBERO env helpers → add a `--env robotwin`
branch or a RoboTwin copy (see §7 note).

================================================================================
## §7. NEW CODE TO WRITE (scaffolds provided; fill SAPIEN/RoboTwin API)
================================================================================
Mirror the LIBERO files 1:1 (`oat/env/libero/env.py`, `factory.py`, `oat/env_runner/libero_runner.py`):
- **`oat/env/robotwin/env.py::RoboTwinEnv`** — wrap RoboTwin's SAPIEN env. MUST expose the exact
  interface the runner/collect use: `reset()`, `step(action)->(obs,reward,done,False,info)`,
  attrs `done`, `cur_step`, `max_episode_steps`, `_extract_obs(raw)->dict` (2 cameras + proprio
  under OAT keys), `render()`, `close()`. Build the env from RoboTwin's task API (TODO: their
  `Env`/`make` call, controller = EEF-delta bimanual → 14D).
- **`oat/env/robotwin/factory.py::get_subtasks(task_name)`** — list of RoboTwin task ids.
- **`oat/env_runner/robotwin_runner.py::RoboTwinRunner`** — COPY `LiberoRunner`, swap
  `LiberoEnv→RoboTwinEnv`, `get_subtasks`. It already calls `policy.predict_action_adaptive(...)`
  and reports `mean_success_rate` → BoN/variable-R work with no runner change.
- **`collect_awr_dataset.py` / `branch_value_k.py`** — they hardcode `from oat.env.libero.env
  import LiberoEnv`; parametrize by `--env` or make a RoboTwin copy for AWR collection (§6).
- **Config templates** (provided): `oat/config/task/tokenizer/robotwin/<task>.yaml`,
  `oat/config/task/policy/robotwin/<task>.yaml` — set `action_dim`, cameras, proprio dims,
  `env_runner._target_: oat.env_runner.robotwin_runner.RoboTwinRunner`, zarr path.

================================================================================
## §8. GOTCHAS (RoboTwin-specific)
================================================================================
- **Action dim (D1)** is the one real model risk — confirm 14 (or set the true value) BEFORE
  retraining the tokenizer. If it includes a base or is joint-space, use that dim consistently.
- **SAPIEN render backend** — RoboTwin uses SAPIEN's GPU renderer (not MUJOCO_GL). The
  `MUJOCO_GL=egl` in the commands above is harmless but SAPIEN has its own headless/EGL setup —
  verify offscreen rendering works on the training box.
- **Isolated env** — SAPIEN/CuRobo/pytorch3d vs robosuite/LIBERO conflict; keep a separate conda
  env (§1).
- **Heavy domain randomization** — RoboTwin randomizes clutter/lighting/background → needs enough
  demos (use the 100K / ≥500 per task) or the policy under-fits → low SR → no BoN headroom.
- **Long-horizon bimanual** — pick a simpler task first; confirm baseline SR is off the floor
  (>~0.2) before BoN/AWR (headroom gate, same as robocasa guide).
- **num_registers change (D2)** propagates `max_seq_len` everywhere via config — don't hardcode 8.

================================================================================
## §9. WHAT TO REPORT BACK (to slot into the paper)
================================================================================
Per task: baseline SR, BoN N∈{1,4,8,16} scaling, AWR single-sample vs base vs BoN-N8, tokenizer
recon-MSE + chosen num_registers, mean episode length / #replans (add a `mean_replans` log line —
needed for the compounding figure F7). This adds a **bimanual + SAPIEN** benchmark → strongest
breadth point ("even dual-arm, a different simulator"), slots into PAPER_MASTER.md §3.3 + §11.
