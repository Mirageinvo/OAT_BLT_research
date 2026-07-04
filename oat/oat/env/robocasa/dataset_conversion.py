"""
Convert RoboCasa (robomimic-format HDF5) demos to the OAT Zarr schema.
Mirror of oat/env/libero/dataset_conversion.py. RoboCasa is robosuite-based; its HDF5 is
robomimic-format: f['data/demo_{i}/obs/<key>'], f['data/demo_{i}/actions'].

The produced Zarr must have EXACTLY these per-step keys (what ZarrDataset + the policy config
expect): action[7], agentview_rgb[H,W,3], robot0_eye_in_hand_rgb[H,W,3], robot0_eef_pos[3],
robot0_eef_quat[4], robot0_gripper_qpos[2], task_uid[1].

=== CRITICAL RoboCasa-specific TODOs — VERIFY against your installed RoboCasa version ===
 1. ACTION SLICING. RoboCasa mobile-manipulation action can be >7D (arm OSC_POSE 6 + gripper 1
    + base 3 + torso/mode). OAT needs EXACTLY 7 (arm+gripper). Set ARM_GRIPPER_IDX to the arm+
    gripper indices. Check the robosuite composite-controller action ordering for your task/
    controller (print demo['actions'].shape and inspect a few rows). If the base actually moves
    in your task, pick a fixed-base task instead (Phase 0 / Р1) — dropping a moving base breaks
    the action-observation consistency.
 2. EEF QUAT. robomimic usually stores 'robot0_eef_quat' directly (xyzw quaternion) -> take
    as-is (NO axisangle2quat, unlike LIBERO). VERIFY the key name + convention.
 3. CAMERA KEYS. RoboCasa image obs keys (e.g. 'robot0_agentview_left_image',
    'robot0_eye_in_hand_image'). Map to OAT's 'agentview_rgb' / 'robot0_eye_in_hand_rgb'.
    Images must be rendered in the dataset (download the *image* dataset variant, or run
    robocasa's dataset_states_to_obs to add camera obs).
 4. IMAGE FLIP. LIBERO flips vertically (np.flip axis=1). RoboCasa may differ -> render one demo
    frame and compare with the LIVE env obs orientation; flip only if mismatched (--flip_images).
 5. GRIPPER QPOS dim. LIBERO gripper_qpos is 2D. Verify RoboCasa's is 2D too (shape_meta says 2).
"""
import h5py
import numpy as np
import tqdm
from typing import Optional, List

from oat.common.replay_buffer import ReplayBuffer

# --- VERIFY these against your RoboCasa HDF5 (print keys + action shape first) ---
ARM_GRIPPER_IDX: List[int] = list(range(0, 7))     # TODO placeholder: first 7 dims. VERIFY!
CAM_AGENTVIEW = 'robot0_agentview_left_image'      # TODO verify key
CAM_EYE = 'robot0_eye_in_hand_image'               # TODO verify key


def convert_robocasa_hdf5_to_zarr(
    hdf5_path: str,
    task_uid: int = 0,
    sample_ndemo: Optional[int] = None,
    flip_images: bool = False,     # set True if the orientation check fails (TODO #4)
) -> ReplayBuffer:
    replay_buffer = ReplayBuffer.create_empty_zarr()
    with h5py.File(hdf5_path, 'r') as f:
        data = f['data']
        demo_keys = sorted(data.keys(), key=lambda k: int(k.split('_')[1]))   # demo_0, demo_1, ...
        if sample_ndemo is not None:
            sel = np.random.choice(len(demo_keys), min(sample_ndemo, len(demo_keys)), replace=False)
            demo_keys = [demo_keys[i] for i in sel]

        for dk in tqdm.tqdm(demo_keys, desc="Converting RoboCasa"):
            demo = data[dk]
            obs = demo['obs']
            demo_len = len(demo['actions'])

            action = demo['actions'][:, ARM_GRIPPER_IDX].astype(np.float32)   # [T, 7]
            assert action.shape[1] == 7, \
                f"action sliced to {action.shape[1]}D, expected 7 — fix ARM_GRIPPER_IDX (TODO #1)"

            av = obs[CAM_AGENTVIEW][:].astype(np.uint8)
            eye = obs[CAM_EYE][:].astype(np.uint8)
            if flip_images:
                av, eye = np.flip(av, axis=1), np.flip(eye, axis=1)

            this = {
                'action': action,
                'agentview_rgb': av,
                'robot0_eye_in_hand_rgb': eye,
                'robot0_eef_pos': obs['robot0_eef_pos'][:].astype(np.float32),
                'robot0_eef_quat': obs['robot0_eef_quat'][:].astype(np.float32),      # xyzw as-is (TODO #2)
                'robot0_gripper_qpos': obs['robot0_gripper_qpos'][:].astype(np.float32),
                'task_uid': np.array([task_uid] * demo_len)[:, np.newaxis],
            }
            replay_buffer.add_episode(this)

    print('-' * 50)
    print(f"{hdf5_path}\n{replay_buffer}")
    return replay_buffer
