# RoboTwin pipeline — just run these in order

All scripts read `config.sh`. **Edit `config.sh` once** (TASK, ROBOTWIN_DIR, NDEMO), then:

```
bash robotwin_pipeline/00_install.sh          # OAT deps + RoboTwin/SAPIEN; then SANITY-check RoboTwin renders
bash robotwin_pipeline/01_download.sh          # demos for TASK
python robotwin_pipeline/02_inspect.py --src data/robotwin_src/<TASK> --format hdf5

### ---- ONE MANUAL STEP (the only code you write) ----
# 02 prints the exact action dim / camera keys / proprio keys. Paste them into:
#   (a) oat/env/robotwin/dataset_conversion.py   (ACTION_IDX, CAM_AGENTVIEW, CAM_EYE, proprio keys)
#   (b) oat/config/task/tokenizer|policy/robotwin/dual_bottles_pick.yaml  (action_dim, shape_meta, camera_names)
#   (c) oat/env/robotwin/env.py                  (fill the SAPIEN reset/step/render TODOs)
#   (d) oat/env_runner/robotwin_runner.py        (copy libero_runner.py + swap LiberoEnv->RoboTwinEnv)
#   (e) add a RoboTwin branch to scripts/collect_awr_dataset.py (only needed for step 09/AWR)
### ---------------------------------------------------

bash   robotwin_pipeline/03_convert.sh          # demos -> Zarr
python robotwin_pipeline/04_verify.py --zarr data/robotwin/<TASK>_N<NDEMO>.zarr --action_dim 14   # GATE
bash   robotwin_pipeline/05_train_tokenizer.sh  # check recon-MSE; if poor: NUM_REGISTERS=16 bash 05...
# -> set TOK_CKPT in config.sh
bash   robotwin_pipeline/06_train_policy.sh
# -> set POLICY_CKPT in config.sh
bash   robotwin_pipeline/07_eval_baseline.sh    # HEADROOM GATE: SR > ~0.2
bash   robotwin_pipeline/08_eval_bon.sh         # BoN sweep (the positive result)
bash   robotwin_pipeline/09_awr.sh              # AWR distillation
```

## Gates (do not skip)
- **After 02:** fill the 5 TODO spots (a–e). Everything downstream depends on it.
- **After 04:** must PASS (correct action dim + both cameras) before training.
- **After 05:** check recon-MSE; bump `num_registers` if poor.
- **After 07:** SR > ~0.2 or switch to a simpler task / more demos.

## What is automated vs manual
- **Automated (just run):** 00 install, 01 download, 03 convert, 04 verify, 05/06 train,
  07/08/09 eval+BoN+AWR — all wrap the reused OAT entry points.
- **Manual (once):** paste 02's output into the converter + configs, fill the SAPIEN env
  (env.py), copy the runner. This is unavoidable — SAPIEN ≠ robosuite. 02_inspect makes (a),(b)
  copy-paste; (c),(d) mirror the LIBERO files exactly.

## Report back (into PAPER_MASTER.md §3.3)
baseline SR, BoN N=4/8/16, AWR single-sample SR, tokenizer recon-MSE + num_registers, mean
episode length / #replans.
