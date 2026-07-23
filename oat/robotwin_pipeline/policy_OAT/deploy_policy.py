"""
OAT policy adapter for the RoboTwin eval harness (path B: use RoboTwin's own tested single-env
eval loop, plug OAT + best-of-N in as a "policy").

COPY this dir to  ~/RoboTwin/policy/OAT/  then run:
  cd ~/RoboTwin/policy/OAT && bash eval.sh pick_dual_bottles demo_clean <ckpt_tag> 0 0
(eval.sh passes the OAT checkpoint path + bon_n via --overrides to get_model.)

RoboTwin obs (live) -> OAT obs keys (matching our converter/config):
  observation["observation"][head_camera]["rgb"]  -> agentview_rgb (resized 128^2)
  observation["observation"][left_camera]["rgb"]  -> robot0_eye_in_hand_rgb
  endpose{left_endpose[7],left_gripper,right_endpose[7],right_gripper} -> agent_pos[16]
Action: OAT decodes the 14D joint chunk -> take_action(a, action_type='qpos').
"""
import os
from collections import deque

import numpy as np
import cv2
import torch

# OAT must be importable (PYTHONPATH includes the oat/ project dir); run with the conda python.
from oat.policy.base_policy import BasePolicy
from oat.policy.oatpolicy import OATPolicy

HEAD_CAM = "head_camera"
WRIST_CAM = "left_camera"
IMG = 128


def _resize(rgb):
    rgb = np.asarray(rgb)
    if rgb.ndim == 3 and rgb.shape[0] in (1, 3):          # CHW -> HWC (just in case)
        rgb = np.transpose(rgb, (1, 2, 0))
    if rgb.dtype != np.uint8:
        rgb = (rgb * 255).clip(0, 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)
    if rgb.shape[:2] != (IMG, IMG):
        rgb = cv2.resize(rgb, (IMG, IMG), interpolation=cv2.INTER_AREA)
    return rgb.astype(np.uint8)


def encode_obs(observation):  # RoboTwin live obs -> OAT obs dict (single frame)
    obs_cams = observation["observation"]
    ep = observation["endpose"]
    agent_pos = np.concatenate([
        np.asarray(ep["left_endpose"], dtype=np.float32).ravel(),
        np.atleast_1d(np.asarray(ep["left_gripper"], dtype=np.float32)).ravel(),
        np.asarray(ep["right_endpose"], dtype=np.float32).ravel(),
        np.atleast_1d(np.asarray(ep["right_gripper"], dtype=np.float32)).ravel(),
    ]).astype(np.float32)                                  # [16]
    return {
        "agentview_rgb": _resize(obs_cams[HEAD_CAM]["rgb"]),
        "robot0_eye_in_hand_rgb": _resize(obs_cams[WRIST_CAM]["rgb"]),
        "agent_pos": agent_pos,
        "task_uid": np.array([0], dtype=np.float32),
    }


class OATModel:
    def __init__(self, ckpt_path, bon_n=1, device="cuda:0", use_k_tokens=8):
        self.policy: OATPolicy = BasePolicy.from_checkpoint(ckpt_path)
        self.policy.to(device).eval()
        assert isinstance(self.policy, OATPolicy)
        self.device = torch.device(device)
        self.dtype = self.policy.dtype
        self.To = self.policy.n_obs_steps
        self.ports = self.policy.get_observation_ports()
        self.bon_n = int(bon_n)
        self.use_k_tokens = use_k_tokens
        self.obs_cache = deque(maxlen=self.To)

    def update_obs(self, obs):
        self.obs_cache.append(obs)

    def reset(self):
        self.obs_cache.clear()

    @torch.inference_mode()
    def get_action(self):
        # build the To-window; left-pad with the earliest frame if not yet full
        frames = list(self.obs_cache)
        while len(frames) < self.To:
            frames.insert(0, frames[0])
        obs_dict = {}
        for port in self.ports:
            stacked = np.stack([f[port] for f in frames[-self.To:]], axis=0)   # [To, ...]
            obs_dict[port] = torch.from_numpy(stacked).to(self.device, self.dtype)[None]  # [1,To,...]
        if self.bon_n > 1:
            res = self.policy.predict_action_adaptive(
                obs_dict, bon_free=self.bon_n, bon_signal="vote", use_k_tokens=self.use_k_tokens)
        else:
            res = self.policy.predict_action_adaptive(
                obs_dict, entropy_threshold=0.0, use_k_tokens=self.use_k_tokens)
        return res["action"][0].detach().cpu().numpy()      # [R, 14]


def get_model(usr_args):
    ckpt = usr_args["oat_ckpt"]                            # from deploy_policy.yml / eval.sh override
    bon_n = int(usr_args.get("bon_n", 1))
    device = usr_args.get("oat_device", "cuda:0")
    print(f"[OAT] loading {ckpt} | bon_n={bon_n}")
    return OATModel(ckpt, bon_n=bon_n, device=device)


def eval(TASK_ENV, model, observation):
    obs = encode_obs(observation)
    if len(model.obs_cache) == 0:
        model.update_obs(obs)
    actions = model.get_action()                          # [R, 14]
    for action in actions:
        TASK_ENV.take_action(action, action_type="qpos")  # 14D joint control
        obs = encode_obs(TASK_ENV.get_obs())
        model.update_obs(obs)


def reset_model(model):
    model.reset()
