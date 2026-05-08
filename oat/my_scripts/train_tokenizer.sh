CUDA_VISIBLE_DEVICES=0 WANDB_MODE=online HYDRA_FULL_ERROR=1 uv run accelerate launch \
	--num_processes 1 \
      scripts/run_workspace.py \
      --config-name=train_oattok \
      task/tokenizer=libero/libero10 \
      training.num_demo=500
