"""
RoboTwin (SAPIEN, Aloha-AgileX bimanual) env wrapper for OAT. Mirror of oat/env/libero/env.py.
Confirmed schema: 14D joint-space action [-1,1]; obs = 3 cameras (head/left/right) + agent_pos[14].
The ONLY thing to fill is the SAPIEN env CONSTRUCTION + reset/step raw-obs plumbing (marked TODO);
everything else (interface, _extract_obs, resize) is done. Two ways to build the underlying env:
  (a) RoboTwin's own task env (their `envs/<task>.py` class), or
  (b) LeRobot's registered robotwin gym env (`--env.type=robotwin`) via lerobot.common.envs.
Pick one in __init__/reset/step (TODO). Keep the returned obs mapping in _extract_obs.
"""
import numpy as np
import gymnasium


def get_subtasks(task_name):
    """RoboTwin task ids. TODO: return the real ROBOTWIN_TASKS list; default = the given one."""
    return [task_name]


def _to_hwc_uint8(img, size):
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    if img.shape[:2] != (size, size):
        try:
            import cv2
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        except Exception:
            from PIL import Image
            img = np.asarray(Image.fromarray(img).resize((size, size)))
    return img.astype(np.uint8)


class RoboTwinEnv(gymnasium.Env):
    def __init__(self, task_name, image_size=128, seed=0,
                 camera_names=('head_camera', 'left_camera'),
                 state_ports=('agent_pos',), max_episode_steps=300,
                 enable_render=True, **kwargs):
        super().__init__()
        self.task_name = task_name
        self.image_size = image_size
        self.camera_names = list(camera_names)
        self.state_ports = list(state_ports)
        self.max_episode_steps = max_episode_steps
        self.enable_render = enable_render
        self.task_uid = 0
        self.done = False
        self.cur_step = 0
        # TODO(build): construct the RoboTwin SAPIEN env for `task_name` with the two cameras and a
        # 14D joint-space controller. e.g. option (b): from lerobot.common.envs.factory import
        # make_env; self.env = make_env('robotwin', task=task_name, camera_names=self.camera_names,
        # observation_height=image_size, observation_width=image_size, seed=seed). Store self.env.
        self.env = None  # TODO

    def _extract_obs(self, raw):
        """RoboTwin raw obs -> OAT dict. raw layout depends on how you built self.env; map here."""
        # TODO map raw -> these keys. Expected raw: images per camera + agent_pos (14D).
        imgs = raw['pixels'] if 'pixels' in raw else raw.get('images', raw)
        obs = {
            'agentview_rgb':          _to_hwc_uint8(imgs[self.camera_names[0]], self.image_size),
            'robot0_eye_in_hand_rgb': _to_hwc_uint8(imgs[self.camera_names[1]], self.image_size),
            'agent_pos':              np.asarray(raw['agent_pos'], dtype=np.float32),
            'task_uid':               np.array([self.task_uid], dtype=np.float32),
        }
        return obs

    def reset(self, seed=None, options=None):
        self.done = False
        self.cur_step = 0
        raw = None  # TODO: obs = self.env.reset(seed=seed); raw = obs
        return self._extract_obs(raw), {}

    def step(self, action):
        """action: [14] joint-space in [-1,1]."""
        self.cur_step += 1
        raw, reward, terminated, info = None, 0.0, False, {}  # TODO: self.env.step(action)
        # TODO: RoboTwin success flag -> reward=1.0 on success (check env.check_success or info).
        self.done = self.done or terminated or (reward >= 1) \
            or (self.cur_step >= self.max_episode_steps)
        return self._extract_obs(raw), reward, self.done, False, info

    def render(self, mode='rgb_array'):
        # TODO: return an RGB frame from a camera for video logging.
        raise NotImplementedError

    def close(self):
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
