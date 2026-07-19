"""Convert official RoboCasa v0.2 HDF5 (human + MimicGen) to OAT zarr.

Paper path (ROBOCASA.md G0): 50 human + 150 machine = 200 demos, Da=12.
Source HDF5: RoboCasa registry human_im / mg_im (demo_gentex_im128_randcams.hdf5).
"""

from __future__ import annotations

import pathlib
from typing import Dict, List, Optional, Sequence

import h5py
import numpy as np
import tqdm
import zarr

from oat.common.replay_buffer import ReplayBuffer

# RoboCasa image keys → OAT zarr names (*_rgb).
OBS_KEY_MAPPING = {
    "robot0_agentview_left_image": "robot0_agentview_left_rgb",
    "robot0_agentview_right_image": "robot0_agentview_right_rgb",
    "robot0_eye_in_hand_image": "robot0_eye_in_hand_rgb",
    "robot0_eef_pos": "robot0_eef_pos",
    "robot0_eef_quat": "robot0_eef_quat",
    "robot0_gripper_qpos": "robot0_gripper_qpos",
    "robot0_base_pos": "robot0_base_pos",
    "robot0_base_quat": "robot0_base_quat",
}

DEFAULT_REQUIRED_OBS_KEYS = (
    "robot0_agentview_left_image",
    "robot0_agentview_right_image",
    "robot0_eye_in_hand_image",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)

EXPECTED_ACTION_DIM = 12
PAPER_N_HUMAN = 50
PAPER_N_MACHINE = 150
PAPER_N_TOTAL = PAPER_N_HUMAN + PAPER_N_MACHINE

# Canonical OAT task ids (snake) ↔ RoboCasa PascalCase dirs.
TASK_ID_TO_PASCAL = {
    "close_drawer": "CloseDrawer",
    "coffee_press_button": "CoffeePressButton",
    "turn_off_microwave": "TurnOffMicrowave",
    "turn_off_sink_faucet": "TurnOffSinkFaucet",
}
PASCAL_TO_TASK_ID = {v: k for k, v in TASK_ID_TO_PASCAL.items()}


def _sorted_demo_keys(data_group: h5py.Group) -> List[str]:
    def demo_sort_key(name: str):
        if name.startswith("demo_"):
            suffix = name[len("demo_") :]
            if suffix.isdigit():
                return (0, int(suffix))
        return (1, name)

    return sorted([k for k in data_group.keys() if k.startswith("demo_")], key=demo_sort_key)


def _select_demo_keys(
    data_group: h5py.Group,
    n: int,
    seed: int,
) -> List[str]:
    demo_keys = _sorted_demo_keys(data_group)
    if not demo_keys:
        raise ValueError("No demos found under 'data'.")
    if n > len(demo_keys):
        raise ValueError(f"Need {n} demos, HDF5 has only {len(demo_keys)}.")
    if n == len(demo_keys):
        return demo_keys
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(demo_keys), n, replace=False)
    idx.sort()
    return [demo_keys[i] for i in idx]


def _to_uint8_image(images: np.ndarray) -> np.ndarray:
    if images.dtype == np.uint8:
        return images
    if np.issubdtype(images.dtype, np.floating):
        maxv = float(np.nanmax(images))
        if maxv <= 1.0:
            images = images * 255.0
    return np.clip(images, 0.0, 255.0).astype(np.uint8)


def _extract_episode(
    demo: h5py.Group,
    required_obs_keys: Sequence[str],
    expected_action_dim: int = EXPECTED_ACTION_DIM,
) -> Dict[str, np.ndarray]:
    if "actions" not in demo:
        raise KeyError("Demo missing 'actions'.")
    if "obs" not in demo:
        raise KeyError("Demo missing 'obs'.")
    action = demo["actions"][:].astype(np.float32)
    if action.ndim != 2 or action.shape[-1] != expected_action_dim:
        raise ValueError(
            f"Expected actions [T, {expected_action_dim}], got {action.shape}."
        )
    obs_group = demo["obs"]
    missing = [k for k in required_obs_keys if k not in obs_group]
    if missing:
        raise KeyError(f"Missing required obs keys: {missing}")

    episode: Dict[str, np.ndarray] = {"action": action}
    for src_key, dst_key in OBS_KEY_MAPPING.items():
        if src_key not in obs_group:
            continue
        values = obs_group[src_key][:]
        if src_key.endswith("_image"):
            values = _to_uint8_image(values)
        elif np.issubdtype(values.dtype, np.floating):
            values = values.astype(np.float32)
        episode[dst_key] = values
    return episode


def convert_robocasa_pair_to_zarr(
    human_hdf5: str,
    mg_hdf5: str,
    zarr_path: str,
    n_human: int = PAPER_N_HUMAN,
    n_machine: int = PAPER_N_MACHINE,
    seed: int = 0,
    required_obs_keys: Sequence[str] = DEFAULT_REQUIRED_OBS_KEYS,
    compressor: Optional[zarr.Blosc] = None,
    chunk_size: int = 1024,
) -> Dict[str, int]:
    """Merge 50 human + 150 MG demos into one zarr (paper N=200)."""
    if compressor is None:
        compressor = zarr.Blosc(cname="zstd", clevel=5, shuffle=1)

    store_path = pathlib.Path(zarr_path)
    if store_path.exists():
        raise FileExistsError(f"{zarr_path} already exists — remove or pick another path.")

    replay_buffer = ReplayBuffer.create_empty_zarr(
        storage=zarr.DirectoryStore(str(store_path)),
    )
    total_steps = 0
    chunks: Optional[Dict[str, tuple]] = None
    sources: List[str] = []

    for label, path, n in (
        ("human", human_hdf5, n_human),
        ("mg", mg_hdf5, n_machine),
    ):
        with h5py.File(path, "r") as f:
            if "data" not in f:
                raise KeyError(f"{path} missing 'data' group.")
            keys = _select_demo_keys(f["data"], n=n, seed=seed)
            sources.append(f"{label}:{path}:n={n}:keys={keys[0]}..{keys[-1]}")
            for demo_key in tqdm.tqdm(keys, desc=f"{label}→zarr"):
                episode = _extract_episode(
                    f["data"][demo_key],
                    required_obs_keys=required_obs_keys,
                )
                if chunks is None:
                    chunks = {
                        k: (chunk_size,) + v.shape[1:] for k, v in episode.items()
                    }
                replay_buffer.add_episode(
                    episode, chunks=chunks, compressors=compressor
                )
                total_steps += int(episode["action"].shape[0])

    if replay_buffer.n_episodes != n_human + n_machine:
        raise RuntimeError(
            f"Expected {n_human + n_machine} episodes, got {replay_buffer.n_episodes}."
        )

    zgroup = zarr.open(str(store_path), mode="r")
    action_dim = int(zgroup["data"]["action"].shape[-1])
    if action_dim != EXPECTED_ACTION_DIM:
        raise RuntimeError(f"action_dim={action_dim}, expected {EXPECTED_ACTION_DIM}.")

    # provenance sidecar
    meta_txt = store_path / "ROBOCASA_SOURCE.txt"
    meta_txt.write_text(
        "\n".join(
            [
                f"n_human={n_human}",
                f"n_machine={n_machine}",
                f"subsample_seed={seed}",
                f"action_dim={action_dim}",
                *sources,
                "",
            ]
        ),
        encoding="utf-8",
    )

    stats = {
        "episodes": int(replay_buffer.n_episodes),
        "steps": int(total_steps),
        "action_dim": action_dim,
        "n_human": n_human,
        "n_machine": n_machine,
    }
    print("-" * 50)
    print(
        f"Saved {zarr_path} | episodes={stats['episodes']} "
        f"(H{n_human}+M{n_machine}) steps={stats['steps']} Da={action_dim}"
    )
    return stats
