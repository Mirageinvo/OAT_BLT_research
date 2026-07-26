# RoboCasa — paper results log (G0+)

**Protocol:** [`ROBOCASA.md`](ROBOCASA.md) (canonical).  
**Also mirrored in:** [`RESULTS.md`](RESULTS.md) § *RoboCasa — Table P*.  
**Claim:** absolute OAT baseline (retrain) + matched Δ (BoN / AWR) on locked ckpts. Report layout = **literal 5 seeds** (not RM/MW `-n 5`).

---

## Seed / HP lock (reproduce)

| Role | Value | ❌ Never |
|------|--------|---------|
| Data mix | 50 human + 150 MG, subsample **seed 0**, Da=12 | regen / LeRobot mirrors |
| Train seed | **0** (tok + policy) | — |
| Selection TopK | `test_start_seed=**2000**`, `n_test=50` → `2000–2049` | paper Table P |
| Paper report | seeds **`10000…10004`**, each `-n 1 --n_test 50` | selection 2000; RM-style `-n 5` @10000 |
| AWR collect | `--seed 0` | 2000* / 10000* |
| **AWR train (BoN-distill)** | **`--epochs 100`**, `--beta 0.5 --beta_kl 0.05` | epochs≠100; RM/MW wave2 defaults |
| Inference | OAT8 + BoN N=8 `vote`; `temp=1.0` `topk=10` | — |
| Venv | `.venv_robocasa` | shared `.venv` |
| New-policy first eval | `rollout_start_epoch=200` (skip ep-0) | — |
| Wave2 script | `cluster_robocasa_literal5_wave2_awr.sh` (EPOCHS=100 hard-fail) | — |

**Aggregation:** mean ± SEM over 5 seed SRs; `SEM_Δ = √(SEM_m²+SEM_b²)`. Artifact: `output/eval/matched_s10000/robocasa/<task>/summary_literal5.json`.

---

## G0 — official data (50H + 150M = 200, Da=12)

| Field | Value |
|-------|--------|
| Source | RoboCasa **v0.2** `dataset_registry.py` `human_im` + `mg_im` (UT Austin Box) |
| Filename | `demo_gentex_im128_randcams.hdf5` |
| Mix | **50 human** (seed-0 subsample from 54) + **150 MimicGen** (seed-0 subsample) |
| Zarr | `data/robocasa/<task>_N200.zarr` |

### Tasks / Box IDs

| task | Pascal | human_im Box id | mg_im Box id |
|------|--------|-----------------|--------------|
| `close_drawer` | CloseDrawer | `4r5w0a6i4jtgv5qmqx09fnqh5d7c45oi` | `aohabqltp8c6ze61u4h2uhtc9p9w35zb` |
| `coffee_press_button` | CoffeePressButton | `l5dnmcfd0r36vhdqgjchxo20vajt7ohl` | `y5zm9mlslfg8p4jkpwnxlizcsvsca1mb` |
| `turn_off_microwave` | TurnOffMicrowave | `0drm2h7fgd5857x8xgcj1lph23srpbj1` | `7c8bku46us9a8sddwg3zk6z3p4ce6ya0` |
| `turn_off_sink_faucet` | TurnOffSinkFaucet | `ceewfn4ydhprupdcdppfe8wu4x61oxdg` | `r392ma0dje2t5ov4dug4vbqhn0z6hblk` |

Base URL: `https://utexas.box.com/shared/static/<id>.hdf5`

### Zarr status + provenance

| task | zarr | size | validate | `ROBOCASA_SOURCE.txt` sha256 |
|------|------|------|----------|------------------------------|
| close_drawer | `close_drawer_N200.zarr` | 1.4G | OK | `b4a8219c740e0f58ca2bd132aa14feaffca3176cca95d268e0f64e821e723c2b` |
| coffee_press_button | `coffee_press_button_N200.zarr` | 809M | OK | `3cd0cfca695c55bb7c3d70e6f4a6a8f7004d0dd99bf8aaf44101691f7734adac` |
| turn_off_microwave | `turn_off_microwave_N200.zarr` | 1.2G | OK | `183855a4c7e315884b8c2d3f7a9a1f569bc33b0d1c1c496b3277db231528747a` |
| turn_off_sink_faucet | `turn_off_sink_faucet_N200.zarr` | 1.2G | OK | `778f8e86853469d81aa4fadcfe97d2c77ab640e03342dbc9b86e54e87e1eb7f8` |

Human HDF5 retained under `data/robocasa/hdf5/<Pascal>/human/` (~240–390M/task). Full/slim MG deleted after convert (disk policy).

**Obs keys:** `action`, `robot0_agentview_{left,right}_rgb`, `robot0_eye_in_hand_rgb`, `robot0_eef_pos/quat`, `robot0_gripper_qpos`.

### G0b success-parity

Still pending as formal gate; policy runs use `SKIP_G0B=1` until PASS logs exist.

---

## Tokenizer (DONE — frozen)

| task | run dir | best MSE ckpt |
|------|---------|---------------|
| close_drawer | `output/20260720/005709_train_oattok_close_drawer_N200` | `checkpoints/ep-1800_mse-0.002.ckpt` |
| coffee_press_button | `output/20260720/041753_train_oattok_coffee_press_button_N200` | `checkpoints/ep-1940_mse-0.003.ckpt` |
| turn_off_microwave | `output/20260720/061925_train_oattok_turn_off_microwave_N200` | `checkpoints/ep-2720_mse-0.002.ckpt` |
| turn_off_sink_faucet | `output/20260720/083055_train_oattok_turn_off_sink_faucet_N200` | `checkpoints/ep-3080_mse-0.002.ckpt` |

Config: `task/tokenizer=robocasa/<task>`, seed 0, batch 256, top-3 by MSE.

---

## Policy train (G1) — status 2026-07-23 ~07:45 MSK

| task | policy run | tmux | status |
|------|------------|------|--------|
| coffee_press_button | `output/20260721/204916_train_oatpolicy_coffee_press_button_N200` | `rc_coffee` + `rc_watch_coffee` | **RESUME** (SIGKILL mid ep-600 eval earlier); TopK sofar **ep-600 @0.28** + `latest` |
| close_drawer | `output/20260723/041317_train_oatpolicy_close_drawer_N200` | `rc_close` + `rc_watch_close` | **RUNNING** (first evals; TopK TBD) |
| turn_off_microwave | *(new dir when starts)* | `rc_microwave` + `rc_watch_microwave` | **queued** (RAM + coffee idle) |
| turn_off_sink_faucet | *(new dir when starts)* | `rc_sink` + `rc_watch_sink` | **queued** (after coffee) |

**Lock → Wave1:** `MIN_EPOCH=3000`, `N_BELOW=4`, `KILL_TRAIN=0`, `TRAIN_END_EPOCH=4500`.  
Lock path: `my_models/robocasa_<task>_topk_lock.txt`.

**Cluster RAM:** ~2 RC policies max in parallel (~14–17 GiB RSS each on 62 GiB host). GPU VRAM not the bottleneck.

### Deleted junk (2026-07-23)

| path | why |
|------|-----|
| `output/20260721/202439_train_oatpolicy_coffee_press_button_N200` | empty hydra, 0 ckpt; superseded by `204916` |
| `output/20260723/041317_train_oatpolicy_turn_off_microwave_N200` | OOM at start, 0 TopK |
| `output/20260723/041317_train_oatpolicy_turn_off_sink_faucet_N200` | killed mid ep-0 for coffee resume, 0 TopK |

**Kept:** all 4 tokenizers; coffee/close **scratch** runs under `20260724/220823_*` (TopK locked 2026-07-26).

---

## Table P (literal-5) — Wave1 RUNNING (locks frozen 2026-07-26)

| task | BASE_CKPT | baseline mean±SEM | BoN | AWR | Δ_BoN±SEM_Δ | Δ_AWR±SEM_Δ | notes |
|------|-----------|-------------------|-----|-----|-------------|-------------|-------|
| close_drawer | `my_models/robocasa_close_drawer_topk_ep0500_sr0.700.ckpt` | *running* | | | | | lock `…_topk_lock.txt` · tmux `rc_wave1_close` · GPU0 |
| coffee_press_button | `my_models/robocasa_coffee_press_button_topk_ep0500_sr0.600.ckpt` | *running* | | | | | lock `…_topk_lock.txt` · tmux `rc_wave1_coffee` · GPU1 |
| turn_off_microwave | TBD | | | | | | deferred |
| turn_off_sink_faucet | TBD | | | | | | deferred |

```bash
SUITE=<task> BASE_CKPT=<locked> GPU=0 bash scripts/cluster_robocasa_literal5_wave1.sh
# → output/eval/matched_s10000/robocasa/<task>/summary_literal5.json
```

**Scratch trains STOPPED 2026-07-26** (user): selection TopK frozen; no further train-evals on these runs.