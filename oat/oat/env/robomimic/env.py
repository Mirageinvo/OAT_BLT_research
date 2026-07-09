import pathlib
from typing import Dict, List, Optional

import gymnasium
import numpy as np
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils

from oat.env.robomimic.env_args_compat import sanitize_env_meta


TASK_NAME_TO_ROBOSUITE_ENV = {
    "lift": "Lift",
    "can": "Can",
    "square": "Square",
}

DEFAULT_HDF5_ROOT = pathlib.Path("data/robomimic/hdf5_datasets")


def resolve_dataset_path(task_name: str, dataset_path: Optional[str] = None) -> str:
    """Resolve RoboMimic HDF5 used for env metadata (prefers multi-human / mh)."""
    task_key = task_name.lower()
    if dataset_path is not None:
        path = pathlib.Path(dataset_path)
        if path.is_file():
            return str(path)
        raise FileNotFoundError(f"RoboMimic dataset not found: {dataset_path}")

    if not DEFAULT_HDF5_ROOT.is_dir():
        raise FileNotFoundError(
            f"RoboMimic HDF5 directory missing: {DEFAULT_HDF5_ROOT}. "
            f"Download mh image data for task '{task_key}'."
        )

    candidates: List[pathlib.Path] = []
    for pattern in (
        f"{task_key}_mh_image.hdf5",
        f"{task_key}_mh*.hdf5",
        f"*{task_key}*mh*image*.hdf5",
        f"*{task_key}*mh*.hdf5",
        f"{task_key}_ph_image.hdf5",
        f"*{task_key}*ph*image*.hdf5",
        f"*{task_key}*.hdf5",
    ):
        candidates.extend(sorted(DEFAULT_HDF5_ROOT.glob(pattern)))

    # de-duplicate while preserving preference order
    seen = set()
    unique_candidates = []
    for path in candidates:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique_candidates.append(path)

    if not unique_candidates:
        raise FileNotFoundError(
            f"No HDF5 found for task '{task_key}' under {DEFAULT_HDF5_ROOT}. "
            "Expected e.g. lift_mh_image.hdf5 (paper uses 200 mh demos)."
        )

    chosen = unique_candidates[0]
    if "mh" not in chosen.stem.lower() and any("mh" in p.stem.lower() for p in unique_candidates):
        chosen = next(p for p in unique_candidates if "mh" in p.stem.lower())

    return str(chosen)


# robosuite 1.5+ metadata fields not accepted by our 1.4 stack (can/square image HDF5).
def _reference_env_meta_for_task(task_key: str) -> dict | None:
    """Lift mh image HDF5 has camera kwargs compatible with our 1.4 stack."""
    if task_key == "lift":
        return None
    lift_h5 = DEFAULT_HDF5_ROOT / "lift_mh_image.hdf5"
    if not lift_h5.is_file():
        return None
    return FileUtils.get_env_metadata_from_dataset(str(lift_h5))


def _init_robomimic_obs_utils(env_kwargs: dict) -> None:
    """robomimic EnvRobosuite.get_observation requires ObsUtils modality map."""
    camera_names = env_kwargs.get("camera_names", [])
    rgb_keys = [
        cam if str(cam).endswith("_image") else f"{cam}_image"
        for cam in camera_names
    ]
    ObsUtils.initialize_obs_modality_mapping_from_dict({
        "rgb": rgb_keys,
        "low_dim": ["object-state"],
    })


def _create_robomimic_env(
    dataset_path: str,
    enable_render: bool,
    seed: int,
    task_key: str = "lift",
):
    raw_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    ref_meta = _reference_env_meta_for_task(task_key)
    env_meta = sanitize_env_meta(raw_meta, reference_env_meta=ref_meta)
    env_kwargs = dict(env_meta.get("env_kwargs", {}))
    _init_robomimic_obs_utils(env_kwargs)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=True,
        use_image_obs=True,
    )
    # Paper: raw RGB as in HDF5 (robomimic create_env defaults to postprocess=True).
    env.postprocess_visual_obs = False
    if hasattr(env, "env") and hasattr(env.env, "seed"):
        env.env.seed(seed)
    return env


class RoboMimicEnv(gymnasium.Env):
    def __init__(
        self,
        task_name: str,
        image_size: int = 84,
        seed: int = 42,
        robot: str = "Panda",
        camera_names: List[str] = ["agentview", "robot0_eye_in_hand"],
        state_ports: List[str] = [
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ],
        video_camera: str = "agentview",
        video_resolution: int = 512,
        max_episode_steps: int = 400,
        control_freq: int = 20,
        enable_render: bool = True,
        dataset_path: Optional[str] = None,
    ):
        super().__init__()
        task_key = task_name.lower()
        if task_key not in TASK_NAME_TO_ROBOSUITE_ENV:
            raise ValueError(
                f"Unsupported RoboMimic task '{task_name}'. "
                f"Supported: {list(TASK_NAME_TO_ROBOSUITE_ENV)}"
            )

        self.task_name = task_key
        self.camera_names = camera_names
        self.state_ports = state_ports
        self.video_camera = video_camera
        self.video_resolution = video_resolution
        self.max_episode_steps = max_episode_steps
        self.control_freq = control_freq
        self.image_size = image_size
        self.dataset_path = resolve_dataset_path(task_key, dataset_path=dataset_path)

        self.env = _create_robomimic_env(
            dataset_path=self.dataset_path,
            enable_render=enable_render,
            seed=seed,
            task_key=task_key,
        )
        self.done = False
        self.cur_step = 0

        obs_dict = self._extract_obs(self.env.reset())
        observation_space = gymnasium.spaces.Dict({})
        for port in self.state_ports:
            if port not in obs_dict:
                raise KeyError(f"State port '{port}' not found in environment observations.")
            observation_space.spaces[port] = gymnasium.spaces.Box(
                low=-np.inf, high=np.inf, shape=obs_dict[port].shape, dtype=np.float32
            )
        for cam_name in self.camera_names:
            observation_space.spaces[f"{cam_name}_rgb"] = gymnasium.spaces.Box(
                low=0, high=255, shape=(image_size, image_size, 3), dtype=np.uint8
            )
        self.observation_space = observation_space
        action_dim = self.env.action_dimension
        self.action_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )

    def _image_key(self, cam_name: str) -> str:
        return f"{cam_name}_image"

    def _extract_obs(self, raw_obs: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
        if raw_obs is None:
            raw_obs = self.env.get_observation()

        obs_dict = {}
        for port in self.state_ports:
            obs_dict[port] = np.asarray(raw_obs[port], dtype=np.float32)
        for cam_name in self.camera_names:
            image_key = self._image_key(cam_name)
            if image_key not in raw_obs:
                raise KeyError(f"Camera obs '{image_key}' missing from environment.")
            # RoboMimic env already applies the same vertical flip as in HDF5 demos.
            obs_dict[f"{cam_name}_rgb"] = np.asarray(raw_obs[image_key], dtype=np.uint8)
        return obs_dict

    def _check_success(self) -> bool:
        if hasattr(self.env, "is_success"):
            succ = self.env.is_success()
            if isinstance(succ, dict):
                return bool(succ.get("task", False))
            return bool(succ)
        inner = getattr(self.env, "env", None)
        if inner is not None and hasattr(inner, "_check_success"):
            return bool(inner._check_success())
        return False

    def step(self, action: np.ndarray):
        obs, _, _, info = self.env.step(action)
        self.cur_step += 1
        success = self._check_success()
        reward = 1.0 if success else 0.0
        self.done = self.done or success or (self.cur_step >= self.max_episode_steps)
        return self._extract_obs(obs), reward, self.done, False, info

    def reset(self, seed=None, options=None):
        if seed is not None and hasattr(self.env, "env") and hasattr(self.env.env, "seed"):
            self.env.env.seed(seed)
        obs = self.env.reset()
        self.done = False
        self.cur_step = 0
        return self._extract_obs(obs), {}

    def render(self, mode="rgb_array"):
        assert mode == "rgb_array"
        frame = self.env.render(
            mode="rgb_array",
            height=self.video_resolution,
            width=self.video_resolution,
            camera_name=self.video_camera,
        )
        return np.asarray(frame, dtype=np.uint8)

    def get_sim_state(self) -> dict:
        if hasattr(self.env, "get_state"):
            return self.env.get_state()
        raise NotImplementedError("Underlying RoboMimic env does not support get_state().")

    def set_state(self, state: dict) -> Dict[str, np.ndarray]:
        if not hasattr(self.env, "reset_to"):
            raise NotImplementedError("Underlying RoboMimic env does not support reset_to().")
        obs = self.env.reset_to(state)
        self.done = False
        self.cur_step = 0
        if obs is None:
            obs = self.env.get_observation()
        return self._extract_obs(obs)

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()
