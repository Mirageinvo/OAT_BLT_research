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

Space note: full `mg_im` ≈ **24.3 GB**/task. Pipeline slim→`*_M150_s0.hdf5` then deletes full MG before convert.

### Status

| task | human HDF5 | mg slim M150 | zarr N200 | validate | sha256 zarr (todo) |
|------|------------|--------------|-----------|----------|--------------------|
| close_drawer | OK (54 demos, 372M) | downloading full MG (~24.3G) in `screen robocasa_g0` | pending | pending | — |
| coffee_press_button | OK (54, 241M) | queued in chain | pending | pending | — |
| turn_off_microwave | OK (54, 389M) | queued in chain | pending | pending | — |
| turn_off_sink_faucet | OK (54, 354M) | queued in chain | pending | pending | — |

**In flight (2026-07-19):** `screen -r robocasa_g0` runs `watch_g0_chain.sh` — one MG at a time → slim M150 (seed 0) → delete full → convert `*_N200.zarr` → next task → `validate_robocasa_data.py`. Monitor: `ls -lh data/robocasa/hdf5/*/mg/` and `logs/g0_*.log`.

Human inspect (2026-07-19): all `actions.shape[-1]==12`; cams `robot0_agentview_{left,right}_image` + `robot0_eye_in_hand_image` @ 128².

### Obs keys in zarr

`action`, `robot0_agentview_left_rgb`, `robot0_agentview_right_rgb`, `robot0_eye_in_hand_rgb`, `robot0_eef_pos`, `robot0_eef_quat`, `robot0_gripper_qpos`, (+ optional `robot0_base_pos/quat`).

Provenance per zarr: `ROBOCASA_SOURCE.txt` inside the zarr directory.

### G0b success-parity

Pending env port — required before policy train (`ROBOCASA.md` §1 G0b).

---

## G1+ (tok / policy / matched)

TBD after G0 green.
