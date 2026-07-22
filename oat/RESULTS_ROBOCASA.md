# RoboCasa — paper results log (G0+)

**Protocol:** `ROBOCASA.md` (canonical).  
**Claim:** matched Δ on fixed ckpts — not Table VI absolute parity.

## G0 — official data (50H + 150M = 200, Da=12)

| Field | Value |
|-------|--------|
| Source | RoboCasa **v0.2** `dataset_registry.py` `human_im` + `mg_im` (UT Austin Box) |
| Filename | `demo_gentex_im128_randcams.hdf5` |
| Mix | **50 human** (seed-0 subsample from 54) + **150 MimicGen** (seed-0 subsample from full mg_im) |
| Subsample seed | **0** (paper collect seed) |
| Action dim | **12** (asserted) |
| Zarr paths | `data/robocasa/<task>_N200.zarr` |

### Tasks / URLs (locked)

| task (snake) | Pascal | human_im Box id | mg_im Box id |
|--------------|--------|-----------------|--------------|
| `close_drawer` | CloseDrawer | `4r5w0a6i4jtgv5qmqx09fnqh5d7c45oi` | `aohabqltp8c6ze61u4h2uhtc9p9w35zb` |
| `coffee_press_button` | CoffeePressButton | `l5dnmcfd0r36vhdqgjchxo20vajt7ohl` | `y5zm9mlslfg8p4jkpwnxlizcsvsca1mb` |
| `turn_off_microwave` | TurnOffMicrowave | `0drm2h7fgd5857x8xgcj1lph23srpbj1` | `7c8bku46us9a8sddwg3zk6z3p4ce6ya0` |
| `turn_off_sink_faucet` | TurnOffSinkFaucet | `ceewfn4ydhprupdcdppfe8wu4x61oxdg` | `r392ma0dje2t5ov4dug4vbqhn0z6hblk` |

Base URL: `https://utexas.box.com/shared/static/<id>.hdf5`

### Pipeline scripts

```bash
# download (human small; mg ≈24GB each — one task at a time)
bash scripts/download_robocasa_paper_hdf5.sh CloseDrawer/human
bash scripts/g0_robocasa_one_task.sh close_drawer   # or watch_g0_chain.sh for all 4
python3 scripts/validate_robocasa_data.py
```

Space note: full `mg_im` ≈ **24 GB**/task. Pipeline: slim→`*_M150_s0.hdf5` → convert → validate → **delete full MG and slim** (retain only `*_N200.zarr`).

### Status

| task | human HDF5 | mg slim M150 | zarr N200 | validate | sha256 zarr (todo) |
|------|------------|--------------|-----------|----------|--------------------|
| close_drawer | OK (54 demos, 372M) | deleted after zarr | **OK** `close_drawer_N200.zarr` (200 eps, Da=12) | **OK** | pending |
| coffee_press_button | OK (54, 241M) | deleted after zarr | **OK** `coffee_press_button_N200.zarr` (200 eps, Da=12) | **OK** | pending |
| turn_off_microwave | OK (54, 389M) | deleted after zarr | **OK** `turn_off_microwave_N200.zarr` (200 eps, Da=12) | **OK** | pending |
| turn_off_sink_faucet | OK (54, 354M) | deleted after zarr | **OK** `turn_off_sink_faucet_N200.zarr` (200 eps, Da=12) | **OK** | pending |

**Disk policy:** full MG → slim M150 → convert N200 → validate → **rm slim** (only zarr retained). CloseDrawer slim removed 2026-07-20 after validate OK.

Human inspect (2026-07-19): all `actions.shape[-1]==12`; cams `robot0_agentview_{left,right}_image` + `robot0_eye_in_hand_image` @ 128².

### Obs keys in zarr

`action`, `robot0_agentview_left_rgb`, `robot0_agentview_right_rgb`, `robot0_eye_in_hand_rgb`, `robot0_eef_pos`, `robot0_eef_quat`, `robot0_gripper_qpos`, (+ optional `robot0_base_pos/quat`).

Provenance per zarr: `ROBOCASA_SOURCE.txt` inside the zarr directory.

### G0b success-parity

Pending env port — required before **policy** train (`ROBOCASA.md` §1 G0b). Does **not** block tokenizer.

### Tokenizer (G0 → tok; parallel OK)

| task | config | train | best MSE ckpt |
|------|--------|-------|---------------|
| close_drawer | `task/tokenizer=robocasa/close_drawer` | **RUNNING** `tmux rc_tok_close_drawer` GPU1; run `output/20260720/005709_train_oattok_close_drawer_N200` (action-only zarr sync ~900K) | TBD |
| coffee_press_button | `task/tokenizer=robocasa/coffee_press_button` | **RUNNING** `tmux rc_tok_coffee_press_button` GPU0; run `output/20260720/041753_train_oattok_coffee_press_button_N200` (action-only ~560K) | TBD |
| turn_off_microwave | `task/tokenizer=robocasa/turn_off_microwave` | **RUNNING** `tmux rc_tok_turn_off_microwave` GPU1; run `output/20260720/061925_train_oattok_turn_off_microwave_N200` | TBD |
| turn_off_sink_faucet | `task/tokenizer=robocasa/turn_off_sink_faucet` | **RUNNING** `tmux rc_tok_turn_off_sink_faucet` GPU1; run `output/20260720/083055_train_oattok_turn_off_sink_faucet_N200` | TBD |

```bash
# on cluster (after rsync zarr + configs):
TASK=close_drawer GPU=1 bash scripts/cluster_tokenizer_robocasa.sh
```

---

## G1+ (tok / policy / matched)

**Eval layout (locked):** literal 5 seeds `10000…10004`, each **one** run `-n 1 --n_test 50`, **same ckpt**, separate dirs `*_seed{seed}/`.

**Aggregation (locked):**
| | |
|--|--|
| cell | `mean ± SEM` over 5 seed SRs (`SEM = SD/√5`; also log SD) |
| Δ_BoN | `mean(BoN) − mean(baseline)`; `SEM_Δ = √(SEM_BoN² + SEM_base²)` |
| Δ_AWR | same vs Wave-1 baseline |
| artifact | `output/eval/matched_s10000/robocasa/<task>/summary_literal5.json` |

Scripts: `cluster_robocasa_literal5_wave1.sh`, `aggregate_robocasa_literal5.py` (see `ROBOCASA.md` §4).  
❌ Not RM/MW `-n 5` on one start seed.

### Table P template (fill after Wave 1/2)

| task | baseline (mean±SEM) | BoN | AWR | Δ_BoN±SEM_Δ | Δ_AWR±SEM_Δ | notes |
|------|---------------------|-----|-----|-------------|-------------|-------|
| close_drawer | | | | | | |
| coffee_press_button | | | | | | |
| turn_off_microwave | | | | | | |
| turn_off_sink_faucet | | | | | | |

Per-seed SRs: paste from `summary_literal5.json` → `methods.*.per_seed_sr`.

TBD after G0 green.
