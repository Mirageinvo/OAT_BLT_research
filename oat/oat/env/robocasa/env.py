"""Gymnasium wrapper for RoboCasa kitchen tasks (CloseDrawer, CoffeePressButton, …).

Requires robosuite >= 1.5 and robocasa v0.2 (use /.venv_robocasa — not shared .venv).
Obs keys match paper zarr (3×128 RGB + eef_pos/quat/gripper_qpos); action Da=12.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import gymnasium
import numpy as np

from oat.env.robocasa.dataset_conversion import TASK_ID_TO_PASCAL

DEFAULT_CAMERA_NAMES = [
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
]
DEFAULT_STATE_PORTS = [
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
]
EXPECTED_ACTION_DIM = 12


def resolve_env_name(task_name: str) -> str:
    key = task_name.strip()
    if key in TASK_ID_TO_PASCAL:
        return TASK_ID_TO_PASCAL[key]
    # Already PascalCase (CloseDrawer, …)
    if key in TASK_ID_TO_PASCAL.values():
        return key
    raise ValueError(
        f"Unknown RoboCasa task '{task_name}'. "
        f"Supported: {list(TASK_ID_TO_PASCAL)}"
    )


def _create_robocasa_env(
    env_name: str,
    seed: int,
    image_size: int,
    camera_names: List[str],
):
    from robocasa.utils.env_utils import create_env

    return create_env(
        env_name=env_name,
        robots="PandaOmron",
        camera_names=camera_names,
        camera_widths=image_size,
        camera_heights=image_size,
        seed=seed,
        render_onscreen=False,
        randomize_cameras=False,
    )


class RoboCasaEnv(gymnasium.Env):
    def __init__(
        self,
        task_name: str,
        image_size: int = 128,
        seed: int = 42,
        camera_names: Optional[List[str]] = None,
        state_ports: Optional[List[str]] = None,
        video_camera: str = "robot0_agentview_left",
        video_resolution: int = 512,
        max_episode_steps: int = 500,
    ):
        super().__init__()
        self.env_name = resolve_env_name(task_name)
        self.task_name = task_name
        self.camera_names = list(camera_names or DEFAULT_CAMERA_NAMES)
        self.state_ports = list(state_ports or DEFAULT_STATE_PORTS)
        self.video_camera = video_camera
        self.video_resolution = video_resolution
        self.max_episode_steps = max_episode_steps
        self.image_size = image_size
        self._seed = int(seed)

        self.env = _create_robocasa_env(
            env_name=self.env_name,
            seed=self._seed,
            image_size=image_size,
            camera_names=self.camera_names,
        )
        self.done = False
        self.cur_step = 0

        raw = self.env.reset()
        if isinstance(raw, tuple):
            raw = raw[0]
        obs_dict = self._extract_obs(raw)

        observation_space = gymnasium.spaces.Dict({})
        for port in self.state_ports:
            if port not in obs_dict:
                raise KeyError(f"State port '{port}' not found in RoboCasa observations.")
            observation_space.spaces[port] = gymnasium.spaces.Box(
                low=-np.inf, high=np.inf, shape=obs_dict[port].shape, dtype=np.float32
            )
        for cam_name in self.camera_names:
            key = f"{cam_name}_rgb"
            observation_space.spaces[key] = gymnasium.spaces.Box(
                low=0, high=255, shape=(image_size, image_size, 3), dtype=np.uint8
            )
        self.observation_space = observation_space

        low, high = self.env.action_spec
        action_dim = int(np.asarray(low).shape[0])
        if action_dim != EXPECTED_ACTION_DIM:
            raise RuntimeError(
                f"Expected action dim {EXPECTED_ACTION_DIM} for RoboCasa, got {action_dim}"
            )
        self.action_space = gymnasium.spaces.Box(
            low=np.asarray(low, dtype=np.float32),
            high=np.asarray(high, dtype=np.float32),
            dtype=np.float32,
        )

    def _image_key(self, cam_name: str) -> str:
        return f"{cam_name}_image"

    def _extract_obs(self, raw_obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        obs_dict: Dict[str, np.ndarray] = {}
        for port in self.state_ports:
            obs_dict[port] = np.asarray(raw_obs[port], dtype=np.float32)
        for cam_name in self.camera_names:
            image_key = self._image_key(cam_name)
            if image_key not in raw_obs:
                raise KeyError(f"Camera obs '{image_key}' missing from RoboCasa env.")
            # Camera obs already match RoboCasa HDF5 / zarr (no extra flip).
            obs_dict[f"{cam_name}_rgb"] = np.asarray(raw_obs[image_key], dtype=np.uint8)
        return obs_dict

    def _check_success(self) -> bool:
        return bool(self.env._check_success())

    def step(self, action: np.ndarray):
        out = self.env.step(action)
        # robosuite 1.5: obs, reward, done, info
        if len(out) == 4:
            obs, _, _, info = out
        else:
            obs, _, _, _, info = out
        self.cur_step += 1
        success = self._check_success()
        reward = 1.0 if success else 0.0
        self.done = self.done or success or (self.cur_step >= self.max_episode_steps)
        return self._extract_obs(obs), reward, self.done, False, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._seed = int(seed)
            if hasattr(self.env, "seed"):
                self.env.seed(self._seed)
        raw = self.env.reset()
        if isinstance(raw, tuple):
            raw = raw[0]
        self.done = False
        self.cur_step = 0
        return self._extract_obs(raw), {}

    def render(self, mode="rgb_array"):
        assert mode == "rgb_array"
        frame = self.env.sim.render(
            height=self.video_resolution,
            width=self.video_resolution,
            camera_name=self.video_camera,
        )[::-1]
        return np.asarray(frame, dtype=np.uint8)

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()
