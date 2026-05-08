CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 HYDRA_FULL_ERROR=1 MUJOCO_GL=egl uv run accelerate launch \
    --num_processes 1 \
    scripts/run_workspace.py \
    --config-name=train_oatpolicy \
    task/policy=libero/libero10 \
    training.rollout_every=500 \
    dataloader.num_workers=16 \
    val_dataloader.num_workers=16 \
    task.policy.lazy_eval=false \
    policy.action_tokenizer.checkpoint=my_models/tokenizer_ep-0820_mse-0.002.ckpt \
    hydra.run.dir=output/20260507/224351_train_oatpolicy_libero10_N500
