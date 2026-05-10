CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 HYDRA_FULL_ERROR=1 MUJOCO_GL=egl uv run accelerate launch \
    --num_processes 1 \
    scripts/run_workspace.py \
    --config-name=train_oatpolicy \
    task/policy=libero/libero10 \
    training.rollout_every=50 \
    task.policy.lazy_eval=false \
    policy.action_tokenizer.checkpoint=my_models/night_models/tokenizer_ep-0950_mse-0.002.ckpt
