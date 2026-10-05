#!/usr/bin/env bash
set -euo pipefail

HF="${HF:-$HOME/.local/bin/hf}"
ROOT="${ROOT:-$HOME/mipt_paper/oat}"
REPO="${REPO:-hackhackhack66666/aaai27-models}"
STAGE="${STAGE:-/tmp/aaai27_models_stage}"
LOG="${LOG:-$HOME/logs/upload_aaai27_models.log}"

mkdir -p "$HOME/logs"
rm -rf "$STAGE"
mkdir -p "$STAGE"/{docs,checkpoints/{my_models,selected_from_output,awr16_from_hf},hydra,eval,replan,logs}

cp -f "$ROOT/docs/aaai27-models/README.md" "$STAGE/README.md"
cp -f "$ROOT/docs/aaai27-models/MANIFEST.json" "$STAGE/MANIFEST.json"

for f in \
  RESULTS.md RESULTS_ROBOCASA.md ROBOCASA.md ROBOMIMIC.md METAWORLD.md \
  METAWORLD_SINGLE_TASK_SPECIALIST.md AGENT_GUIDE_TABLE_C_PRIME_LATENCY.md \
  AGENT_GUIDE_AWR16_LATENCY_REMAINING.md AGENT_GUIDE_ROBOMIMIC_TABLEP_N250.md
do
  cp -f "$ROOT/$f" "$STAGE/docs/$f"
done

cp -al "$ROOT/my_models/." "$STAGE/checkpoints/my_models/"
cp -al "$ROOT/output/eval/matched_s10000" "$STAGE/eval/matched_s10000"
cp -al "$ROOT/eval_out" "$STAGE/eval/eval_out"
cp -al "$ROOT/logs/." "$STAGE/logs/"

while read -r rel; do
  [[ -n "$rel" ]] || continue
  mkdir -p "$STAGE/checkpoints/selected_from_output/$(dirname "$rel")"
  cp -al "$ROOT/$rel" "$STAGE/checkpoints/selected_from_output/$rel"
done <<'LIST'
output/20260705/210939_train_oattok_can_N200/checkpoints/ep-0520_mse-0.005.ckpt
output/20260704/203215_train_oattok_lift_N200/checkpoints/ep-1970_mse-0.006.ckpt
output/20260706/005048_train_oattok_square_N200/checkpoints/ep-0690_mse-0.004.ckpt
output/20260710/212943_train_oattok_mw-coffee-pull_st_N50/checkpoints/ep-2670_mse-0.039.ckpt
output/20260710/235437_train_oattok_mw-stick-pull_st_N50/checkpoints/ep-3030_mse-0.042.ckpt
output/20260710/235437_train_oattok_mw-disassemble_st_N50/checkpoints/ep-3410_mse-0.027.ckpt
output/20260710/212943_train_oattok_mw-box-close_st_N50/checkpoints/ep-3450_mse-0.019.ckpt
output/20260720/005709_train_oattok_close_drawer_N200/checkpoints/ep-1800_mse-0.002.ckpt
output/20260720/041753_train_oattok_coffee_press_button_N200/checkpoints/ep-1940_mse-0.003.ckpt
output/20260720/061925_train_oattok_turn_off_microwave_N200/checkpoints/ep-2720_mse-0.002.ckpt
output/20260720/083055_train_oattok_turn_off_sink_faucet_N200/checkpoints/ep-3080_mse-0.002.ckpt
output/20260707/124135_train_oattok_mw-mt4_N50/checkpoints/ep-3830_mse-0.024.ckpt
output/20260706/173343_train_oatpolicy_can_N200/checkpoints/ep-1700_sr-0.940.ckpt
output/20260719/144024_train_oatpolicy_lift_N200/checkpoints/ep-0900_sr-0.930.ckpt
output/20260719/144024_train_oatpolicy_lift_N200/checkpoints/ep-1400_sr-0.950.ckpt
output/20260720/215024_train_oatpolicy_square_N200/checkpoints/ep-0700_sr-0.420.ckpt
output/20260720/090816_train_oatpolicy_mw-coffee-pull_st_N50/checkpoints/ep-1000_sr-0.432.ckpt
output/20260711/134439_train_oatpolicy_mw-stick-pull_st_N50/checkpoints/ep-0800_sr-0.212.ckpt
output/20260711/134440_train_oatpolicy_mw-disassemble_st_N50/checkpoints/ep-1400_sr-0.700.ckpt
output/20260711/134439_train_oatpolicy_mw-box-close_st_N50/checkpoints/ep-2000_sr-0.552.ckpt
output/20260724/220823_train_oatpolicy_close_drawer_N200/checkpoints/ep-0500_sr-0.700.ckpt
output/20260724/220823_train_oatpolicy_coffee_press_button_N200/checkpoints/ep-0500_sr-0.600.ckpt
output/20260725/231926_train_oatpolicy_turn_off_sink_faucet_N200/checkpoints/ep-0500_sr-0.580.ckpt
output/20260725/233258_train_oatpolicy_turn_off_microwave_N200/checkpoints/ep-0500_sr-0.620.ckpt
LIST

for run in \
  output/20260704/203215_train_oattok_lift_N200 \
  output/20260705/210939_train_oattok_can_N200 \
  output/20260706/005048_train_oattok_square_N200 \
  output/20260706/173343_train_oatpolicy_can_N200 \
  output/20260707/124135_train_oattok_mw-mt4_N50 \
  output/20260710/212943_train_oattok_mw-box-close_st_N50 \
  output/20260710/212943_train_oattok_mw-coffee-pull_st_N50 \
  output/20260710/235437_train_oattok_mw-disassemble_st_N50 \
  output/20260710/235437_train_oattok_mw-stick-pull_st_N50 \
  output/20260711/134439_train_oatpolicy_mw-box-close_st_N50 \
  output/20260711/134439_train_oatpolicy_mw-stick-pull_st_N50 \
  output/20260711/134440_train_oatpolicy_mw-disassemble_st_N50 \
  output/20260719/144024_train_oatpolicy_lift_N200 \
  output/20260720/005709_train_oattok_close_drawer_N200 \
  output/20260720/041753_train_oattok_coffee_press_button_N200 \
  output/20260720/061925_train_oattok_turn_off_microwave_N200 \
  output/20260720/083055_train_oattok_turn_off_sink_faucet_N200 \
  output/20260720/090816_train_oatpolicy_mw-coffee-pull_st_N50 \
  output/20260720/215024_train_oatpolicy_square_N200 \
  output/20260724/220823_train_oatpolicy_close_drawer_N200 \
  output/20260724/220823_train_oatpolicy_coffee_press_button_N200 \
  output/20260725/231926_train_oatpolicy_turn_off_sink_faucet_N200 \
  output/20260725/233258_train_oatpolicy_turn_off_microwave_N200
do
  if [[ -d "$ROOT/$run/.hydra" ]]; then
    mkdir -p "$STAGE/hydra/$run"
    cp -al "$ROOT/$run/.hydra" "$STAGE/hydra/$run/.hydra"
  fi
done

# Recover available AWR16 checkpoints from Mirageinv/AWR
mkdir -p "$STAGE/checkpoints/awr16_from_hf"
for f in \
  robomimic_can_awr_bon16_e100.ckpt \
  robomimic_lift_awr_bon16_e100.ckpt \
  robomimic_square_awr_bon16_e100.ckpt \
  robocasa_coffee_press_button_awr_bon16_e100.ckpt
do
  "$HF" download Mirageinv/AWR "$f" --local-dir "$STAGE/checkpoints/awr16_from_hf" >/dev/null
done

find "$STAGE" -type f | sort > "$STAGE/INVENTORY.txt"
echo "[$(date -Is)] stage summary" | tee "$LOG"
du -sh "$STAGE"/* | tee -a "$LOG"

"$HF" upload-large-folder "$REPO" "$STAGE" --repo-type model --num-workers 4 2>&1 | tee -a "$LOG"
