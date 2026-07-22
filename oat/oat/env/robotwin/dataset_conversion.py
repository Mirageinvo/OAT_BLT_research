"""
Convert RoboTwin 2.0 demos -> OAT Zarr schema. Mirror of oat/env/libero/dataset_conversion.py.
RoboTwin `lerobot/robotwin_unified` is a single embodiment (Aloha-AgileX): action=14D joint-space
[-1,1], obs=pixels_agent_pos (3 cameras + observation.state[14]). Two source formats:
  - LeRobot v3.0 (HF Hub, recommended) -> convert_robotwin_lerobot_to_zarr  (concrete below)
  - native per-episode HDF5 (RoboTwin data-collection) -> convert_robotwin_hdf5_to_zarr

Produced Zarr per-step keys: action[14], agentview_rgb[128,128,3], robot0_eye_in_hand_rgb[128,128,3],
agent_pos[14], task_uid[1].

VERIFY (run robotwin_pipeline/02_inspect.py first) the exact LeRobot feature key names below.
"""
import numpy as np
import tqdm
from typing import Optional

from oat.common.replay_buffer import ReplayBuffer

# --- LeRobot feature keys (verify with 02_inspect) ---
LR_ACTION = 'action'                                   # [14]
LR_STATE = 'observation.state'                         # agent_pos [14]
LR_CAM_AGENTVIEW = 'observation.images.head_camera'    # -> agentview_rgb
LR_CAM_EYE = 'observation.images.left_camera'          # -> robot0_eye_in_hand_rgb
ACTION_DIM = 14


def _to_hwc_uint8(img, size):
    """LeRobot image (torch CHW float[0,1] | np HWC | PIL) -> HWC uint8 resized to (size,size)."""
    import numpy as np
    try:
        import torch
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
            if img.ndim == 3 and img.shape[0] in (1, 3):      # CHW -> HWC
                img = np.transpose(img, (1, 2, 0))
            if img.dtype != np.uint8:
                img = (img * 255).clip(0, 255).astype(np.uint8)
    except Exception:
        pass
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    if img.shape[:2] != (size, size):                          # resize 240x320 -> size^2
        try:
            import cv2
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        except Exception:
            from PIL import Image
            img = np.asarray(Image.fromarray(img).resize((size, size)))
    return img.astype(np.uint8)


def convert_robotwin_lerobot_to_zarr(src: str, task_uid: int = 0,
                                     sample_ndemo: Optional[int] = None,
                                     flip_images: bool = False,
                                     image_size: int = 128) -> ReplayBuffer:
    """LeRobot v3.0 dataset (local dir or HF repo id). Requires the `lerobot` package."""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset   # verify import path
    rb = ReplayBuffer.create_empty_zarr()
    ds = LeRobotDataset(src)
    # episode boundaries: LeRobotDataset exposes episode_data_index {'from':[..],'to':[..]}
    edi = getattr(ds, 'episode_data_index', None)
    n_ep = ds.num_episodes if hasattr(ds, 'num_episodes') else len(edi['from'])
    if sample_ndemo is not None:
        n_ep = min(n_ep, sample_ndemo)

    for ep in tqdm.tqdm(range(n_ep), desc='Converting RoboTwin (lerobot)'):
        lo = int(edi['from'][ep]); hi = int(edi['to'][ep])
        act, av, eye, state = [], [], [], []
        for t in range(lo, hi):
            fr = ds[t]
            act.append(np.asarray(fr[LR_ACTION], dtype=np.float32)[:ACTION_DIM])
            state.append(np.asarray(fr[LR_STATE], dtype=np.float32))
            a = _to_hwc_uint8(fr[LR_CAM_AGENTVIEW], image_size)
            e = _to_hwc_uint8(fr[LR_CAM_EYE], image_size)
            if flip_images:
                a, e = np.flip(a, 0), np.flip(e, 0)
            av.append(a); eye.append(e)
        T = len(act)
        rb.add_episode({
            'action': np.stack(act),
            'agent_pos': np.stack(state),
            'agentview_rgb': np.stack(av),
            'robot0_eye_in_hand_rgb': np.stack(eye),
            'task_uid': np.array([task_uid] * T)[:, np.newaxis],
        })
    print('-' * 50); print(f"{src}\n{rb}")
    return rb


def convert_robotwin_hdf5_to_zarr(src_dir: str, task_uid: int = 0,
                                  sample_ndemo: Optional[int] = None,
                                  flip_images: bool = False,
                                  image_size: int = 128) -> ReplayBuffer:
    """Native per-episode HDF5 under src_dir/*.hdf5. VERIFY keys with 02_inspect (D1/D3/D4)."""
    import h5py, glob, os
    rb = ReplayBuffer.create_empty_zarr()
    files = sorted(glob.glob(os.path.join(src_dir, '**', '*.hdf5'), recursive=True))
    if sample_ndemo is not None:
        files = files[:sample_ndemo]
    for f in tqdm.tqdm(files, desc='Converting RoboTwin (hdf5)'):
        with h5py.File(f, 'r') as h:
            # TODO adjust to RoboTwin's HDF5 layout (02_inspect prints the real keys/shapes).
            action = np.asarray(h['action'][:, :ACTION_DIM], dtype=np.float32)
            state = np.asarray(h['observation/agent_pos'][:], dtype=np.float32)
            av = np.stack([_to_hwc_uint8(x, image_size) for x in h['observation/head_camera'][:]])
            eye = np.stack([_to_hwc_uint8(x, image_size) for x in h['observation/left_camera'][:]])
            if flip_images:
                av, eye = np.flip(av, 1), np.flip(eye, 1)
            T = action.shape[0]
            rb.add_episode({
                'action': action, 'agent_pos': state,
                'agentview_rgb': av, 'robot0_eye_in_hand_rgb': eye,
                'task_uid': np.array([task_uid] * T)[:, np.newaxis],
            })
    print('-' * 50); print(f"{src_dir}\n{rb}")
    return rb
