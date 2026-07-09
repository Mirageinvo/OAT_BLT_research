"""
Generate MetaWorld expert demonstrations (oracle policy) into Zarr for OAT training.

Paper (Appendix A): MT4 = box-close, coffee-pull, disassemble, stick-pull;
50 successful demos per task.

Usage:
  cd oat
  MUJOCO_GL=egl uv run python scripts/gen_metaworld_data.py \\
    --task_name mt4 --num_episodes 50 --device cuda:0

Output:
  data/metaworld/mt4_N50.zarr  # 50 demos per task, 200 episodes total for MT4
"""
if __name__ == "__main__":
    import os
    import pathlib
    import sys

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import copy
import os
import pathlib
from typing import List, Tuple

import click
import numpy as np
import zarr
import metaworld.policies

from oat.common.input_util import wait_user_input
from oat.common.replay_buffer import ReplayBuffer
from oat.env.metaworld.factory import get_metaworld_env, get_subtasks


def load_expert_policy(task_name: str) -> List:
    subtask_names = get_subtasks(task_name)
    experts = []
    for name in subtask_names:
        if name == "peg-insert-side":
            expert_name = "PegInsertionSide"
        else:
            expert_name = "".join(s.capitalize() for s in name.split("-"))
        expert_cls = getattr(
            metaworld.policies,
            f"Sawyer{expert_name}V2Policy",
            None,
        )
        if expert_cls is None:
            expert_cls = getattr(metaworld.policies, f"Sawyer{expert_name}V3Policy")
        experts.append(expert_cls())
    return experts


@click.command()
@click.option("-t", "--task_name", type=str, required=True, help="mt4 | mt10 | single task slug")
@click.option("-d", "--root_data_dir", type=str, default="data/metaworld")
@click.option("-s", "--save_dir", type=str, default=None)
@click.option(
    "-c",
    "--num_episodes",
    type=int,
    default=50,
    help="successful episodes per task for multitask suites (total for a single task)",
)
@click.option(
    "-o",
    "--sensors",
    multiple=True,
    type=str,
    default=("corner", "corner2", "corner3", "behindGripper"),
    show_default=True,
)
@click.option("--device", type=str, default="cuda:0")
@click.option("--save_subtasks", is_flag=True, help="also write per-subtask zarr shards")
def gen_metaworld_data(
    task_name: str,
    root_data_dir: str,
    save_dir: str,
    num_episodes: int,
    sensors: Tuple[str, ...],
    device: str,
    save_subtasks: bool,
):
    if save_dir is None:
        # N follows the paper convention: demos per task for multitask suites.
        save_dir = os.path.join(root_data_dir, f"{task_name}_N{num_episodes}.zarr")

    save_paths = [save_dir]
    if os.path.exists(save_dir):
        keypress = wait_user_input(
            valid_input=lambda key: key in ["", "y", "n"],
            prompt=f"{save_dir} already exists. Overwrite? [y/`n`]: ",
            default="n",
        )
        if keypress == "n":
            print("Abort")
            return
        os.system(f"rm -rf {save_dir}")
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

    envs = get_metaworld_env(
        task_name=task_name,
        image_size=128,
        camera_names=list(sensors),
        device=device,
        oracle=True,
    )
    experts = load_expert_policy(task_name)
    assert len(envs) == len(experts)
    assert not save_subtasks or len(envs) > 1

    if save_subtasks:
        per_task = num_episodes
        for senv in envs:
            sdir = os.path.join(root_data_dir, f"{senv.task_name}_N{per_task}.zarr")
            if os.path.exists(sdir):
                keypress = wait_user_input(
                    valid_input=lambda key: key in ["", "y", "n"],
                    prompt=f"{sdir} already exists. Overwrite? [y/`n`]: ",
                    default="n",
                )
                if keypress == "n":
                    print("Abort")
                    return
                os.system(f"rm -rf {sdir}")
            pathlib.Path(sdir).mkdir(parents=True)
            save_paths.append(sdir)

    num_envs = len(envs)
    total_episodes = num_episodes * num_envs
    env_indices = [i % num_envs for i in range(total_episodes)]

    replay_buffers = [
        ReplayBuffer.create_empty_zarr()
        for _ in range((num_envs + 1) if save_subtasks else 1)
    ]
    data_dtype = {f"{sensor}_rgb": "uint8" for sensor in sensors}
    accepted_counts = np.zeros(num_envs, dtype=np.int64)

    episode_idx = 0
    while episode_idx < total_episodes:
        env_idx = env_indices[episode_idx]
        env = envs[env_idx]
        expert = experts[env_idx]

        obs_dict, _ = env.reset()
        done = False
        episode_reward = 0.0
        episode_success = False
        episode_success_count = 0

        this_data_collected = {"action": []}
        while not done:
            action = expert.get_action(obs_dict["full_state"])

            for k, v in obs_dict.items():
                if k not in this_data_collected:
                    this_data_collected[k] = []
                this_data_collected[k].append(v)
            this_data_collected["action"].append(action)

            obs_dict, reward, done, _, info = env.step(action)
            episode_reward += reward
            episode_success = episode_success or bool(info.get("success", False))
            episode_success_count += int(bool(info.get("success", False)))

        if not episode_success:
            print(
                f"Episode {episode_idx + 1}, task={env.task_name}, failed "
                f"(reward={episode_reward:.3f}, success_count={episode_success_count})"
            )
            continue

        episode_idx += 1
        for key in this_data_collected:
            this_data_collected[key] = np.array(
                this_data_collected[key], dtype=data_dtype.get(key, "float32")
            )
        replay_buffers[-1].add_episode(copy.deepcopy(this_data_collected))
        accepted_counts[env_idx] += 1
        if save_subtasks:
            replay_buffers[env_idx].add_episode(copy.deepcopy(this_data_collected))
        print(
            f"Episode {episode_idx}/{total_episodes}, task={env.task_name}, "
            f"reward={episode_reward:.1f}, success_count={episode_success_count}"
        )

    replay_buffers[-1].update_meta({
        "subtask_counts": accepted_counts,
        "num_episodes_per_task": np.array(num_episodes, dtype=np.int64),
    })

    for buff, sdir in zip(replay_buffers, save_paths):
        print("-" * 50)
        print(f"{sdir}:\n{buff}")

    compressor = zarr.Blosc(cname="zstd", clevel=5, shuffle=1)
    for buff, sdir in zip(replay_buffers, save_paths):
        buff.save_to_path(sdir, compressors=compressor)

    for env in envs:
        env.close()


if __name__ == "__main__":
    gen_metaworld_data()
