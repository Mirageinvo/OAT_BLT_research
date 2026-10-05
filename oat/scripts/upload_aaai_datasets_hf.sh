#!/usr/bin/env bash
# Upload paper HDF5+Zarr to hackhackhack66666/aaai-datasets (resumable, no local copy).
# Run on cluster HOST (not docker). Needs: hf CLI + token (hackhackhack66666).
set -euo pipefail

HF="${HF:-$HOME/.local/bin/hf}"
REPO="hackhackhack66666/aaai-datasets"
OAT="${OAT:-$HOME/mipt_paper/oat}"
DOCS="${OAT}/docs/aaai-datasets"
STAGE="${STAGE:-/tmp/aaai_datasets_stage}"
LOG="${LOG:-$HOME/logs/upload_aaai_datasets.log}"

mkdir -p "$HOME/logs" "$STAGE"/{robomimic/{zarr,hdf5},metaworld/zarr,robocasa/{zarr,hdf5}}

echo "[$(date -Is)] staging symlinks -> $STAGE" | tee -a "$LOG"

# docs (real files)
cp -f "$DOCS/README.md" "$STAGE/README.md"
cp -f "$DOCS/MANIFEST.json" "$STAGE/MANIFEST.json"
cp -f "$DOCS/METAWORLD_GENERATION.md" "$STAGE/metaworld/METAWORLD_GENERATION.md"

link() { ln -sfn "$1" "$2"; }

# RoboMimic zarr
for t in lift can square; do
  link "$OAT/data/robomimic/${t}_N200.zarr" "$STAGE/robomimic/zarr/${t}_N200.zarr"
done
# RoboMimic hdf5
link "$OAT/data/robomimic/hdf5_datasets/lift_mh_image.hdf5" "$STAGE/robomimic/hdf5/lift_mh_image.hdf5"
link "$OAT/data/robomimic/hdf5_datasets/can_mh_image.hdf5" "$STAGE/robomimic/hdf5/can_mh_image.hdf5"
link "$OAT/data/robomimic/hdf5_datasets/square_mh_image.hdf5" "$STAGE/robomimic/hdf5/square_mh_image.hdf5"
mkdir -p "$STAGE/robomimic/hdf5/can/mh"
link "$OAT/data/robomimic/hdf5_datasets/can/mh/demo_v15.hdf5" "$STAGE/robomimic/hdf5/can/mh/demo_v15.hdf5"
link "$OAT/data/robomimic/hdf5_datasets/can/mh/image_v15.hdf5" "$STAGE/robomimic/hdf5/can/mh/image_v15.hdf5"

# MetaWorld zarr
for z in mt4_N50 box-close_N50 coffee-pull_N50 disassemble_N50 stick-pull_N50; do
  link "$OAT/data/metaworld/${z}.zarr" "$STAGE/metaworld/zarr/${z}.zarr"
done

# RoboCasa zarr
for t in close_drawer coffee_press_button turn_off_sink_faucet turn_off_microwave; do
  link "$OAT/data/robocasa/${t}_N200.zarr" "$STAGE/robocasa/zarr/${t}_N200.zarr"
done
# RoboCasa hdf5 (partial — MG deleted on cluster after convert)
mkdir -p "$STAGE/robocasa/hdf5/CloseDrawer/human" "$STAGE/robocasa/hdf5/CoffeePressButton/human"
if [[ -f "$OAT/data/robocasa/hdf5/CloseDrawer/human/demo_gentex_im128_randcams.hdf5" ]]; then
  link "$OAT/data/robocasa/hdf5/CloseDrawer/human/demo_gentex_im128_randcams.hdf5" \
    "$STAGE/robocasa/hdf5/CloseDrawer/human/demo_gentex_im128_randcams.hdf5"
fi
if [[ -f "$OAT/data/robocasa/hdf5/CoffeePressButton/human/demo_gentex_im128_randcams.hdf5" ]]; then
  link "$OAT/data/robocasa/hdf5/CoffeePressButton/human/demo_gentex_im128_randcams.hdf5" \
    "$STAGE/robocasa/hdf5/CoffeePressButton/human/demo_gentex_im128_randcams.hdf5"
fi

echo "[$(date -Is)] du stage:" | tee -a "$LOG"
du -sh "$STAGE"/* | tee -a "$LOG"

echo "[$(date -Is)] upload-large-folder -> $REPO" | tee -a "$LOG"
"$HF" upload-large-folder "$REPO" "$STAGE" \
  --repo-type dataset \
  --num-workers 4 \
  2>&1 | tee -a "$LOG"

echo "[$(date -Is)] DONE" | tee -a "$LOG"
