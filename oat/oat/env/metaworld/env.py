import warnings
from typing import Dict, List, Optional, Union

import gymnasium
import numpy as np

warnings.filterwarnings("ignore", message=".*Box bound precision lowered by casting to float32")

try:
    from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE as MW_ENV_REGISTRY

    MW_ENV_VERSION = 2
except ImportError:
    from metaworld.env_dict import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE as MW_ENV_REGISTRY

    MW_ENV_VERSION = 3


class MetaworldEnv(gymnasium.Env):
    """MetaWorld v2 goal-observable env with multi-camera RGB + proprio (OAT paper setup)."""

    metadata = {
        "render_modes": ["rgb_array"],
        "video.frames_per_second": 10,
    }

    def __init__(
        self,
        task_name: str,
        device: str = "cuda:0",
        image_size: int = 128,
        seed: Optional[int] = None,
        camera_names: Optional[List[str]] = None,
        oracle: bool = False,
        video_camera: str = "corner2",
        video_resolution: int = 512,
        max_episode_steps: int = 200,
        enable_render: bool = True,
    ):
        super().__init__()
        if camera_names is None:
            camera_names = [
                "topview",
                "corner",
                "corner2",
                "corner3",
                "behindGripper",
                "gripperPOV",
            ]

        env_key = f"{task_name}-v{MW_ENV_VERSION}-goal-observable"
        self.env = MW_ENV_REGISTRY[env_key](seed=seed, render_mode="rgb_array")
        if seed is not None:
            self.env.seed(seed)
        if hasattr(self.env, "_freeze_rand_vec"):
            self.env._freeze_rand_vec = not oracle

        self._reset_inner_env()
        self.env_init_state = self.env.get_env_state()

        model = self._model
        model.vis.map.znear = 0.1
        model.vis.map.zfar = 1.5

        # corner2 camera tweak (https://arxiv.org/abs/2212.05698)
        cam_id = self._camera_name2id("corner2")
        assert cam_id == 2
        if hasattr(model, "cam_pos0"):
            model.cam_pos0[cam_id] = [0.6, 0.295, 0.8]
        model.cam_pos[cam_id] = [0.6, 0.295, 0.8]

        self.camera_names = camera_names
        self.episode_length = max_episode_steps
        self.max_episode_steps = max_episode_steps
        self.oracle = oracle
        self.image_size = image_size
        self.video_resolution = video_resolution
        self.task_name = task_name
        self.video_camera = video_camera
        self.enable_render = enable_render
        self.gpu_id = int(device.split(":")[-1]) if enable_render else 0

        self.done = False
        self.cur_step = 0
        self.succ_step = 0

        observation_space = gymnasium.spaces.Dict()
        for name in camera_names:
            observation_space.spaces[f"{name}_rgb"] = gymnasium.spaces.Box(
                low=0,
                high=255,
                shape=(image_size, image_size, 3),
                dtype=np.uint8,
            )
        agent_pos = np.concatenate(
            [
                self.env.get_endeff_pos(),
                self.env._get_site_pos("leftEndEffector"),
                self.env._get_site_pos("rightEndEffector"),
            ]
        )
        observation_space.spaces["agent_pos"] = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=agent_pos.shape,
            dtype=np.float32,
        )
        if oracle:
            observation_space.spaces["full_state"] = gymnasium.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=self.env.observation_space.shape,
                dtype=np.float32,
            )
        self.observation_space = observation_space
        self.action_space = gymnasium.spaces.Box(
            low=self.env.action_space.low,
            high=self.env.action_space.high,
            shape=self.env.action_space.shape,
            dtype=np.float32,
        )

    @property
    def _model(self):
        return self.env.sim.model if hasattr(self.env, "sim") else self.env.model

    def _camera_name2id(self, camera_name: str) -> int:
        model = self._model
        if hasattr(model, "camera_name2id"):
            return model.camera_name2id(camera_name)
        for i in range(model.ncam):
            if model.camera(i).name == camera_name:
                return i
        raise KeyError(camera_name)

    def _reset_inner_env(self):
        reset_out = self.env.reset()
        return reset_out[0] if isinstance(reset_out, tuple) else reset_out

    def _get_rgb(
        self,
        cam_name: Optional[List[str]] = None,
        image_size: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        if cam_name is None:
            cam_name = self.camera_names
        if image_size is None:
            image_size = self.image_size
        if hasattr(self.env, "sim"):
            return {
                cam: self.env.sim.render(
                    width=image_size,
                    height=image_size,
                    camera_name=cam,
                    depth=False,
                    device_id=self.gpu_id,
                )
                for cam in cam_name
            }

        renderer = self.env.mujoco_renderer
        old_width = renderer.width
        old_height = renderer.height
        old_camera_id = renderer.camera_id
        try:
            renderer.width = image_size
            renderer.height = image_size
            images = {}
            for cam in cam_name:
                renderer.camera_id = self._camera_name2id(cam)
                images[cam] = self.env.render()
            return images
        finally:
            renderer.width = old_width
            renderer.height = old_height
            renderer.camera_id = old_camera_id

    def _extract_obs(self, full_state: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        obs_dict = {
            "agent_pos": np.concatenate(
                [
                    self.env.get_endeff_pos(),
                    self.env._get_site_pos("leftEndEffector"),
                    self.env._get_site_pos("rightEndEffector"),
                ]
            ).astype(np.float32)
        }
        for cam_name, rgb in self._get_rgb().items():
            obs_dict[f"{cam_name}_rgb"] = rgb
        if self.oracle and full_state is not None:
            obs_dict["full_state"] = np.asarray(full_state, dtype=np.float32)
        return obs_dict

    def step(self, action: np.ndarray):
        step_out = self.env.step(action)
        if len(step_out) == 5:
            full_state, _reward, terminated_inner, truncated_inner, info = step_out
            _done = terminated_inner or truncated_inner
        else:
            full_state, _reward, _done, info = step_out
        self.cur_step += 1
        obs_dict = self._extract_obs(full_state if self.oracle else None)
        if info.get("success"):
            self.succ_step += 1
        success = bool(info.get("success", False))
        reward = 1.0 if success else 0.0
        terminated = success or (self.succ_step >= 10)
        truncated = self.cur_step >= self.episode_length
        self.done = terminated or truncated
        return obs_dict, reward, terminated or truncated, False, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.env.seed(seed)
            self.env_init_state = self._env_init_state_of_seed(seed)
        self._reset_inner_env()
        if hasattr(self.env, "reset_model"):
            self.env.reset_model()
        full_state = self._reset_inner_env()
        self.env.set_env_state(self.env_init_state)
        self.cur_step = 0
        self.succ_step = 0
        self.done = False
        obs_dict = self._extract_obs(full_state if self.oracle else None)
        return obs_dict, {}

    def render(self, mode="rgb_array"):
        assert mode == "rgb_array"
        return self._get_rgb([self.video_camera], image_size=self.video_resolution)[
            self.video_camera
        ]

    def close(self):
        self.env.close()

    def _env_init_state_of_seed(self, seed: int):
        env = MW_ENV_REGISTRY[f"{self.task_name}-v{MW_ENV_VERSION}-goal-observable"](
            seed=seed,
            render_mode="rgb_array",
        )
        reset_out = env.reset()
        init_state = env.get_env_state()
        env.close()
        return init_state
