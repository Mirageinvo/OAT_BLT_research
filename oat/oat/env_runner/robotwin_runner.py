"""
RoboTwinRunner — a verbatim copy of LiberoRunner with LiberoEnv -> RoboTwinEnv and RoboTwin
defaults (agent_pos state, head/left cameras, 300-step episodes). The run() loop is unchanged
(it is env-agnostic: operates on the wrapped env via MultiStep/VideoRecording/AsyncVectorEnv),
so BoN and variable-R work with no change. The ONLY RoboTwin-specific code left to fill is the
SAPIEN env construction inside oat/env/robotwin/env.py::RoboTwinEnv.
"""
import wandb
import numpy as np
import torch
import tqdm
import math
import pathlib
import dill
import wandb.sdk.data_types.video as wandb_video

from oat.gymnasium_util.multistep_wrapper import MultiStepWrapper
from oat.gymnasium_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from oat.gymnasium_util.async_vector_env import AsyncVectorEnv
from oat.env.robotwin.env import RoboTwinEnv, get_subtasks
from oat.env_runner.base_runner import BaseRunner
from oat.policy.base_policy import BasePolicy
from oat.common.pytorch_util import dict_apply

from typing import Optional, List


def maybe_to_torch(x, device, dtype):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device, dtype=dtype)
    return x


class RoboTwinRunner(BaseRunner):
    def __init__(self,
        output_dir,
        task_name: str,
        n_test: int,
        n_test_vis: int,
        test_start_seed: int = 1000,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        fps: int = 25,
        crf: int = 22,
        tqdm_interval_sec: float = 5.0,
        n_parallel_envs: Optional[int] = None,
        image_size: int = 128,
        camera_names: List[str] = ["head_camera", "left_camera"],
        state_ports: List[str] = ["agent_pos"],
        max_episode_steps: int = 300,
    ):
        super().__init__(output_dir)

        if n_parallel_envs is None:
            n_parallel_envs = n_test
        n_parallel_envs = min(n_parallel_envs, n_test)
        assert n_parallel_envs > 0, "n_parallel_envs must be positive"
        assert n_test_vis <= n_test, "n_test_vis must be <= n_test"

        subtask_names = get_subtasks(task_name)
        num_tasks = len(subtask_names)
        num_repeats = math.ceil(n_test / num_tasks)
        env_task_names = []
        for batch_start in range(0, num_tasks, n_parallel_envs):
            batch_tasks = subtask_names[batch_start:batch_start + n_parallel_envs]
            for _ in range(num_repeats):
                env_task_names.extend(batch_tasks)
        env_task_names = env_task_names[:n_test]

        env_seeds, env_fns, env_init_fn_dills = [], [], []
        for i in range(n_test):
            this_task_name = env_task_names[i]
            this_seed = test_start_seed + i
            env_seeds.append(this_seed)
            enable_render = i < n_test_vis

            if i < n_parallel_envs:
                def env_fn(task_name=this_task_name, seed=this_seed):
                    return MultiStepWrapper(
                        VideoRecordingWrapper(
                            RoboTwinEnv(
                                task_name=task_name, image_size=image_size, seed=seed,
                                camera_names=camera_names, state_ports=state_ports,
                                max_episode_steps=max_episode_steps,
                            ),
                            video_recoder=VideoRecorder.create_h264(
                                fps=fps, codec='h264', input_pix_fmt='rgb24', crf=crf,
                                thread_type='FRAME', thread_count=1),
                            file_path=None, steps_per_render=1,
                        ),
                        n_obs_steps=n_obs_steps, n_action_steps=n_action_steps,
                        max_episode_steps=max_episode_steps, reward_agg_method='max',
                    )
                env_fns.append(env_fn)

            def init_fn(env, task_name=this_task_name, seed=this_seed, enable_render=enable_render):
                if env.env.env.task_name != task_name:
                    env.env.env.close()
                    env.env.env = RoboTwinEnv(
                        task_name=task_name, image_size=image_size, seed=seed,
                        camera_names=camera_names, state_ports=state_ports,
                        max_episode_steps=max_episode_steps,
                    )
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        f'media/{task_name}', wandb_video.util.generate_id() + '.mp4')
                    filename.parent.mkdir(parents=True, exist_ok=True)
                    env.env.file_path = str(filename)
                env.reset()
            env_init_fn_dills.append(dill.dumps(init_fn))

        assert len(env_fns) == n_parallel_envs
        assert len(env_init_fn_dills) == n_test

        def dummy_env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RoboTwinEnv(
                        task_name=env_task_names[0], image_size=image_size,
                        camera_names=camera_names, state_ports=state_ports,
                        max_episode_steps=max_episode_steps, enable_render=False,
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps, codec='h264', input_pix_fmt='rgb24', crf=crf,
                        thread_type='FRAME', thread_count=1),
                    file_path=None, steps_per_render=1,
                ),
                n_obs_steps=n_obs_steps, n_action_steps=n_action_steps,
                max_episode_steps=max_episode_steps, reward_agg_method='max',
            )

        env = AsyncVectorEnv(env_fns, shared_memory=False, dummy_env_fn=dummy_env_fn)

        self.env = env
        self.task_name = task_name
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_task_names = env_task_names
        self.env_init_fn_dills = env_init_fn_dills
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_episode_steps = max_episode_steps
        self.tqdm_interval_sec = tqdm_interval_sec

    @torch.inference_mode()
    def run(self, policy: BasePolicy, **kwargs):
        device = policy.device
        dtype = policy.dtype
        policy_name = policy.get_policy_name()

        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        all_video_paths = [None] * n_inits
        all_success = [False] * n_inits
        episode_done = np.zeros(n_inits, dtype=bool)
        all_token_counts, all_k_preds, all_r_execs = [], [], []

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0, this_n_active_envs)

            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]] * n_diff)
            assert len(this_init_fns) == n_envs

            self.env.call_each('run_dill_function', args_list=[(x,) for x in this_init_fns])
            obs, _ = self.env.reset()
            policy.reset()

            pbar = tqdm.tqdm(
                total=self.max_episode_steps,
                desc=f"Eval {policy_name} in RoboTwin::{self.task_name} {chunk_idx+1}/{n_chunks}",
                leave=False, mininterval=self.tqdm_interval_sec)

            done = False
            while not done and pbar.n < pbar.total:
                obs_dict = dict_apply(obs, lambda x: maybe_to_torch(x, device=device, dtype=dtype))
                with torch.inference_mode():
                    result = policy.predict_action_adaptive(
                        {port: obs_dict[port] for port in policy.get_observation_ports()}, **kwargs)
                    action = result['action'].detach().cpu().numpy()
                    if 'n_tokens' in result:
                        all_token_counts.append(result['n_tokens'])
                    if 'k_pred' in result:
                        all_k_preds.append(result['k_pred'].detach().cpu().numpy())

                r_exec = None
                if 'r_exec' in result:
                    r_exec = result['r_exec'].detach().cpu().numpy().astype(int)
                    all_r_execs.append(r_exec)
                    action = np.array(action, copy=True)
                    for i in range(action.shape[0]):
                        action[i, int(r_exec[i]):, :] = np.nan

                if r_exec is None:
                    if not np.all(np.isfinite(action)):
                        raise RuntimeError("NaN or Inf action")
                else:
                    for i in range(action.shape[0]):
                        if not np.all(np.isfinite(action[i, :int(r_exec[i])])):
                            raise RuntimeError("NaN or Inf action")

                obs, reward, env_done, _, _ = self.env.step(action)

                env_done_local = np.asarray(env_done[this_local_slice], dtype=bool)
                succ_local = np.asarray([r >= 1 for r in reward[this_local_slice]], dtype=bool)
                ep_done_local = episode_done[start:end]
                first_ep = ~ep_done_local
                cur_success = np.asarray(all_success[this_global_slice], dtype=bool)
                all_success[this_global_slice] = (cur_success | (succ_local & first_ep)).tolist()
                episode_done[start:end] = ep_done_local | env_done_local | succ_local
                done = bool(np.all(episode_done[start:end]))

                step_advance = int(r_exec.min()) if r_exec is not None else action.shape[1]
                pbar.update(step_advance)
            pbar.close()
            all_video_paths[this_global_slice] = self.env.render()[this_local_slice]

        _ = self.env.reset()

        log_data = dict()
        for task_name in set(self.env_task_names):
            task_success = [all_success[i] for i in range(n_inits)
                            if self.env_task_names[i] == task_name]
            log_data[f"{task_name}/mean_success_rate"] = np.mean(task_success)
        for i in range(n_inits):
            video_path = all_video_paths[i]
            if video_path is not None:
                log_data[f"{self.env_task_names[i]}/video_{self.env_seeds[i]}"] = \
                    wandb.Video(video_path, format='mp4')
        log_data['mean_success_rate'] = np.mean(all_success)
        if all_token_counts:
            log_data['mean_tokens_used'] = np.mean(all_token_counts)
        # mean replans/episode (for the compounding figure): 1 policy call per step here
        if all_token_counts:
            log_data['mean_replans'] = len(all_token_counts) / max(1, n_inits)
        if all_r_execs:
            rs = np.concatenate([np.atleast_1d(r).ravel() for r in all_r_execs]).astype(int)
            log_data['mean_r_exec'] = float(rs.mean())
        return log_data

    def close(self):
        self.env.close()
