import wandb
import numpy as np
import torch
import tqdm
import math
import time
import hashlib
import pathlib
import dill
import wandb.sdk.data_types.video as wandb_video

from oat.gymnasium_util.multistep_wrapper import MultiStepWrapper
from oat.gymnasium_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from oat.gymnasium_util.async_vector_env import AsyncVectorEnv
from oat.env.libero.env import LiberoEnv
from oat.env.libero.factory import get_subtasks
from oat.env_runner.base_runner import BaseRunner
from oat.policy.base_policy import BasePolicy
from oat.common.pytorch_util import dict_apply

from typing import Optional, List

def _scene_hash(env):
    """md5 of the MuJoCo state right after reset (env = MultiStepWrapper in the worker).
    Used to verify that every variant starts from identical initial scenes."""
    try:
        st = env.env.env.env.get_sim_state()          # LiberoEnv.env = ControlEnv
        st = np.round(np.asarray(st, dtype=np.float64), 8)
        return hashlib.md5(st.tobytes()).hexdigest()[:16]
    except Exception as e:                             # never break the eval for a diagnostic
        return f"err:{type(e).__name__}"


def _n_env_steps(env):
    """Executed env steps of the worker's current episode (call before the worker
    autoresets, i.e. right after the step on which the episode ended)."""
    return len(env.get_rewards())


def _n_resets(env):
    """LiberoEnv reset counter; used to detect a hidden worker-side reset between the
    explicit reset of a chunk and its first executed step."""
    return int(getattr(env.env.env, 'n_resets', -1))


def maybe_to_torch(x, device, dtype):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device, dtype=dtype)
    else:
        return x
    
class LiberoRunner(BaseRunner):
    # import multiprocessing as mp
    # mp.set_start_method('spawn', force=True)

    def __init__(self,
        output_dir,
        task_name: str,
        n_test: int,
        n_test_vis: int,
        test_start_seed: int = 1000,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        fps: int = 20,
        crf: int = 22,
        tqdm_interval_sec: float = 5.0,
        n_parallel_envs: Optional[int] = None,
        # other libero env args
        image_size: int = 128,
        camera_names: List[str] = [
            "agentview",
            "robot0_eye_in_hand",
        ],
        state_ports: List[str] = [
            'robot0_joint_pos',
            'robot0_eef_pos',
            'robot0_eef_quat',
            'robot0_gripper_qpos',
        ],
        max_episode_steps: int = 550,
    ):
        super().__init__(output_dir)

        if n_parallel_envs is None:
            n_parallel_envs = n_test
        n_parallel_envs = min(n_parallel_envs, n_test)

        assert n_parallel_envs > 0, "n_parallel_envs must be positive"
        assert n_test_vis <= n_test, "n_test_vis must be less than or equal to n_test"

        # get subtasks and distribute across envs in batches
        subtask_names = get_subtasks(task_name)
        num_tasks = len(subtask_names)
        num_repeats = math.ceil(n_test / num_tasks)
        env_task_names = []
        for batch_start in range(0, num_tasks, n_parallel_envs):
            batch_tasks = subtask_names[batch_start:batch_start + n_parallel_envs]
            for _ in range(num_repeats):
                env_task_names.extend(batch_tasks)
        env_task_names = env_task_names[:n_test]

        # setup env
        env_seeds = []
        env_fns = []
        env_init_fn_dills = []
        for i in range(n_test):
            this_task_name = env_task_names[i]
            this_seed = test_start_seed + i
            env_seeds.append(this_seed)
            enable_render = i < n_test_vis

            if i < n_parallel_envs:
                def env_fn(task_name=this_task_name, seed=this_seed):
                    return MultiStepWrapper(
                        VideoRecordingWrapper(
                            LiberoEnv(
                                task_name=task_name,
                                image_size=image_size,
                                seed=seed,
                                camera_names=camera_names,
                                state_ports=state_ports,
                                max_episode_steps=max_episode_steps,
                            ),
                            video_recoder=VideoRecorder.create_h264(
                                fps=fps,
                                codec='h264',
                                input_pix_fmt='rgb24',
                                crf=crf,
                                thread_type='FRAME',
                                thread_count=1
                            ),
                            file_path=None,
                            steps_per_render=1
                        ),
                        n_obs_steps=n_obs_steps,
                        n_action_steps=n_action_steps,
                        max_episode_steps=max_episode_steps,
                        reward_agg_method='max'
                    )
                env_fns.append(env_fn)
        
            def init_fn(env, task_name=this_task_name, seed=this_seed, enable_render=enable_render):
                if env.env.env.task_name != task_name:
                    env.env.env.close()
                    env.env.env = LiberoEnv(
                        task_name=task_name,
                        image_size=image_size,
                        seed=seed,
                        camera_names=camera_names,
                        state_ports=state_ports,
                        max_episode_steps=max_episode_steps,
                    )
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        f'media/{task_name}', wandb_video.util.generate_id() + '.mp4'
                    )
                    filename.parent.mkdir(parents=True, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename
                env.reset()
            env_init_fn_dills.append(dill.dumps(init_fn))

        assert len(env_fns) == n_parallel_envs
        assert len(env_init_fn_dills) == n_test

        # For each process the OpenGL context can only be initialized once
        # Since AsyncVectorEnv uses fork to create worker process,
        # a separate env_fn that does not create OpenGL context (enable_render=False)
        # is needed to initialize spaces.
        def dummy_env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    LiberoEnv(
                        task_name=env_task_names[0],
                        image_size=image_size,
                        # seed=seed,
                        camera_names=camera_names,
                        state_ports=state_ports,
                        max_episode_steps=max_episode_steps,
                        enable_render=False,
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=1
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_episode_steps,
                reward_agg_method='max'
            )

        env = AsyncVectorEnv(env_fns, shared_memory=False,
            dummy_env_fn=dummy_env_fn
            # context='spawn',
        )  # NOTE: turn off shared_memory to use Text space

        # attr assignment
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
    def run(self, 
        policy: BasePolicy,
        # policy inference args
        **kwargs,
    ):
        device = policy.device
        dtype = policy.dtype
        policy_name = policy.get_policy_name()

        # ---- AAMAS-27 rebuttal logging (popped: not policy kwargs) ----
        # gen_seed: if set, torch RNG is re-seeded at the start of every env batch with
        #   f(gen_seed, first global episode index). Batch membership is fixed by
        #   (n_test, n_parallel_envs), and the policy is queried for all batch rows every
        #   cycle (finished envs are frozen, not reset), so the generation RNG stream of a
        #   batch does not depend on how long previous batches lasted.
        gen_seed = kwargs.pop('gen_seed', None)
        log_diagnostics = bool(kwargs.pop('log_diagnostics', False))
        if log_diagnostics:
            kwargs['return_candidates'] = True
        cs_fixed_h = kwargs.get('cs_fixed_h', None)
        use_cuda = (torch.device(device).type == 'cuda')
        self.last_episode_rows = []
        self.last_decision_rows = []

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_success = [False] * n_inits
        # per-init: has its FIRST episode ended? Needed because variable-R desyncs the
        # envs -> a fast env can finish & autoreset into a 2nd episode while slow envs
        # are still running; we must not count that 2nd episode (would inflate SR).
        episode_done = np.zeros(n_inits, dtype=bool)
        all_token_counts = []
        all_k_preds = []
        all_r_execs = []

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

            # init envs
            self.env.call_each(
                'run_dill_function', 
                args_list=[(x,) for x in this_init_fns]
            )

            # start rollout. The FINAL reset of every episode is seeded with its label
            # (10000+i), so the initial scene depends only on (task, seed label), not on
            # how many resets the worker did before (env reuse across chunks / repeats).
            reset_seeds = list(self.env_seeds[this_global_slice])
            reset_seeds += [self.env_seeds[0]] * (n_envs - len(reset_seeds))
            obs, _ = self.env.reset(seed=reset_seeds)
            policy.reset()
            scene_hashes = self.env.call_each(
                'run_dill_function', args_list=[(dill.dumps(_scene_hash),)] * n_envs)
            resets_at_start = self.env.call_each(
                'run_dill_function', args_list=[(dill.dumps(_n_resets),)] * n_envs)
            hidden_reset = np.zeros(this_n_active_envs, dtype=bool)
            ep_steps = np.full(this_n_active_envs, -1, dtype=int)
            first_step = True
            if gen_seed is not None:
                torch.manual_seed((int(gen_seed) * 1_000_003 + start) % (2 ** 62))
            calls_active = np.zeros(this_n_active_envs, dtype=int)
            infer_ms_active = np.zeros(this_n_active_envs, dtype=float)

            pbar = tqdm.tqdm(
                total=self.max_episode_steps,
                desc=f"Eval {policy_name} in Libero::{self.task_name} {chunk_idx+1}/{n_chunks}",
                leave=False,
                mininterval=self.tqdm_interval_sec
            )

            done = False
            while not done and pbar.n < pbar.total:
                active_before = ~episode_done[start:end]
                # ---- timed region: obs -> device, perception, generation, decode, select,
                # chunk back on host. Excludes env.step, diagnostics, logging. ----
                if use_cuda:
                    torch.cuda.synchronize(device)
                _tic = time.perf_counter()
                # create obs dict
                obs_dict = dict_apply(
                    obs,
                    lambda x: maybe_to_torch(x, device=device, dtype=dtype)
                )

                # run policy
                with torch.inference_mode():
                    result = policy.predict_action_adaptive({
                        port: obs_dict[port]
                        for port in policy.get_observation_ports()
                    }, **kwargs)
                    action = result['action'].detach().cpu().numpy()
                    if use_cuda:
                        torch.cuda.synchronize(device)
                    call_ms = (time.perf_counter() - _tic) * 1000.0
                    # ---- end timed region ----
                    calls_active += active_before.astype(int)
                    infer_ms_active += active_before * call_ms
                    if log_diagnostics and 'bon_cand' in result:
                        diag = policy.bon_diagnostics(
                            result['bon_cand'][:this_n_active_envs], result['bon_R'],
                            cs_fixed_h=cs_fixed_h)
                        sel = result['selected_idx_b'][:this_n_active_envs].tolist()
                        for j in range(this_n_active_envs):
                            if not active_before[j]:
                                continue
                            gi = start + j
                            self.last_decision_rows.append(dict(
                                task=self.env_task_names[gi], init_id=gi,
                                seed_label=self.env_seeds[gi], repeat_seed=gen_seed,
                                replan_id=int(calls_active[j]) - 1, R=int(result['bon_R']),
                                executed_idx=int(sel[j]), **diag[j]))
                    if 'n_tokens' in result:
                        all_token_counts.append(result['n_tokens'])
                    if 'k_pred' in result:
                        all_k_preds.append(result['k_pred'].detach().cpu().numpy())

                # variable-R (GATE 1): NaN-pad each env's chunk beyond its executed
                # length R so MultiStepWrapper stops there; envs then advance at
                # different sim-rates (independent episodes -> fine).
                r_exec = None
                if 'r_exec' in result:
                    r_exec = result['r_exec'].detach().cpu().numpy().astype(int)
                    all_r_execs.append(r_exec)
                    action = np.array(action, copy=True)   # ensure writable
                    for i in range(action.shape[0]):
                        action[i, int(r_exec[i]):, :] = np.nan

                if r_exec is None:
                    if not np.all(np.isfinite(action)):
                        raise RuntimeError("NaN of Inf action")
                else:
                    # only the executed (non-sentinel) prefix must be finite
                    for i in range(action.shape[0]):
                        if not np.all(np.isfinite(action[i, :int(r_exec[i])])):
                            raise RuntimeError("NaN or Inf action")

                # step env
                obs, reward, env_done, _, _ = self.env.step(action) # NOTE: reward=1 if success
                if first_step:
                    # the first step must execute the action in the scene we hashed, not
                    # trigger a pending autoreset (would discard the action + change scene)
                    r_now = self.env.call_each(
                        'run_dill_function', args_list=[(dill.dumps(_n_resets),)] * n_envs)
                    hidden_reset = np.array(
                        [r_now[j] != resets_at_start[j] for j in range(this_n_active_envs)])
                    first_step = False

                # first-episode bookkeeping (variable-R desync safety, see episode_done).
                # success is counted only while the env is still in its first episode;
                # once an env's first episode ends it is frozen so its autoreset 2nd
                # episode cannot change the result. Reduces to the original lockstep
                # behaviour when all envs finish on the same cycle (fixed-R).
                env_done_local = np.asarray(env_done[this_local_slice], dtype=bool)
                succ_local = np.asarray([r >= 1 for r in reward[this_local_slice]], dtype=bool)
                ep_done_local = episode_done[start:end]
                first_ep = ~ep_done_local
                cur_success = np.asarray(all_success[this_global_slice], dtype=bool)
                all_success[this_global_slice] = (cur_success | (succ_local & first_ep)).tolist()
                newly_done = first_ep & (env_done_local | succ_local)
                if newly_done.any():
                    # read step counts now: the worker autoresets on the next step, after
                    # which get_rewards() would describe the 2nd (ignored) episode
                    st = self.env.call_each(
                        'run_dill_function', args_list=[(dill.dumps(_n_env_steps),)] * n_envs)
                    for j in np.flatnonzero(newly_done):
                        ep_steps[j] = int(st[j])
                episode_done[start:end] = ep_done_local | env_done_local | succ_local
                done = bool(np.all(episode_done[start:end]))

                # update pbar by the slowest env's progress so we never cut a slow
                # (small-R) env short before it reaches its episode end
                step_advance = int(r_exec.min()) if r_exec is not None else action.shape[1]
                pbar.update(step_advance)
            pbar.close()

            env_steps = self.env.call_each(
                'run_dill_function', args_list=[(dill.dumps(_n_env_steps),)] * n_envs)
            for j in range(this_n_active_envs):
                gi = start + j
                steps_j = int(ep_steps[j]) if ep_steps[j] >= 0 else int(env_steps[j])
                status = 'ok'
                if hidden_reset[j]:
                    status = 'hidden_reset'
                elif str(scene_hashes[j]).startswith('err:'):
                    status = 'scene_hash_error'
                self.last_episode_rows.append(dict(
                    task=self.env_task_names[gi], init_id=gi, seed_label=self.env_seeds[gi],
                    repeat_seed=gen_seed, success=int(bool(all_success[gi])),
                    episode_finished=int(bool(episode_done[gi])),
                    env_steps=steps_j, policy_calls=int(calls_active[j]),
                    inference_ms=float(infer_ms_active[j]), batch_envs=int(n_envs),
                    scene_hash=scene_hashes[j], status=status))

            # collect data for this round
            all_video_paths[this_global_slice] = self.env.render()[this_local_slice]

        # clear out video buffer
        _ = self.env.reset()

        # log
        log_data = dict()

        for task_name in set(self.env_task_names):
            task_success = [
                all_success[i]
                for i in range(n_inits)
                if self.env_task_names[i] == task_name
            ]
            log_data[f"{task_name}/mean_success_rate"] = np.mean(task_success)

        for i in range(n_inits):
            seed = self.env_seeds[i]
            task_name = self.env_task_names[i]

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                video = wandb.Video(video_path, format='mp4')
                log_data[f"{task_name}/video_{seed}"] = video
            
        # log aggregate metrics
        log_data['mean_success_rate'] = np.mean(all_success)
        if all_token_counts:
            log_data['mean_tokens_used'] = np.mean(all_token_counts)

        # in-sim distribution of per-sample token budget (predictor / agnostic-mix modes)
        if all_k_preds:
            ks = np.concatenate([np.atleast_1d(k).ravel() for k in all_k_preds]).astype(int)
            maxk = int(ks.max())
            hist = np.bincount(ks, minlength=maxk + 1)[1:]
            fracs = hist / hist.sum()
            print(f"[k_pred] in-sim mean={ks.mean():.3f} | hist(k=1..{maxk})={hist.tolist()} | "
                  f"fracs={[round(float(x), 3) for x in fracs]}")
            # scalar fractions survive eval_policy_sim's numeric aggregation
            for kk in range(1, maxk + 1):
                log_data[f'k_pred_frac_{kk}'] = float((ks == kk).mean())

        # in-sim distribution of per-sample executed chunk length R (variable-R mode).
        # mean_r_exec is the mean replan interval -> replan/vision-CNN cost ~ 1/mean_r_exec
        if all_r_execs:
            rs = np.concatenate([np.atleast_1d(r).ravel() for r in all_r_execs]).astype(int)
            log_data['mean_r_exec'] = float(rs.mean())
            lo, hi = int(rs.min()), int(rs.max())
            hist = np.bincount(rs, minlength=hi + 1)[lo:]
            print(f"[r_exec] in-sim mean={rs.mean():.3f} | range=[{lo},{hi}] | "
                  f"hist(R={lo}..{hi})={hist.tolist()}")

        return log_data

    def close(self):
        self.env.close()
