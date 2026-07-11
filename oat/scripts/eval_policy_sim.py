"""
Usage:
python experiments/eval_policy_sim.py --checkpoint path/to/ckpt -o path/to/output_dir
"""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import wandb
import json
import numpy as np
from oat.env_runner.base_runner import BaseRunner
from oat.policy.base_policy import BasePolicy
from oat.model.token_count_predictor import TokenCountPredictor
from typing import List, Optional

@click.command()
@click.option('-c', '--checkpoint', required=True, help="either a .ckpt file or a directory containing .ckpt files")
@click.option('-o', '--output_dir', required=True, help="output directory for eval info dump")
@click.option('-n', '--num_exp', default=1, help="num experiments to run")
@click.option('-d', '--device', default='cuda:0', help="device to run on")
@click.option('--temperature', default=None, type=float, help="temperature for policy inference")
@click.option('--topk', default=None, type=int, help="topk for policy inference")
@click.option('--use_k_tokens', default=None, type=int, help="number of tokens to use for policy inference")
@click.option('--token_predictor', default=None, type=str,
              help="path to a TokenCountPredictor .ckpt; if set, adaptive generation uses it "
                   "instead of the entropy threshold")
@click.option('--entropy_threshold', default=None, type=float,
              help="entropy threshold for entropy-mode early stopping (default 2.75). "
                   "Set <=0 to disable early stopping (full budget k=max_seq_len). "
                   "Ignored when --token_predictor is set.")
@click.option('--agnostic_mix', default=None, type=str,
              help="obs-agnostic budget mixture baseline, e.g. '1:0.09,2:0.21,8:0.68'. "
                   "Samples k from this categorical per sample (ignores obs). "
                   "Takes precedence over --token_predictor.")
@click.option('--n_action_steps', default=None, type=int,
              help="override executed chunk length R (steps executed open-loop before replanning; "
                   "default 16, max = decode horizon 32). For the fixed-R sweep.")
@click.option('--adaptive_r', default=None, type=str,
              help="variable-R execution mode (GATE 1): 'convergence' | 'random' | 'fixed'. "
                   "Holds K at use_k_tokens (default full) and adapts the executed chunk length "
                   "R per observation. Forces n_action_steps = r_max.")
@click.option('--r_coarse_k', default=4, type=int,
              help="coarse token budget for the convergence R-signal (decode_k vs decode_K)")
@click.option('--r_min', default=8, type=int, help="min executed chunk length R")
@click.option('--r_max', default=32, type=int, help="max executed chunk length R (<= horizon 32)")
@click.option('--r_threshold', default=0.5, type=float,
              help="convergence divergence threshold (normalizer-space L2); execute the leading "
                   "prefix where coarse and full plans agree below this")
@click.option('--bon_free', default=0, type=int,
              help="verifier-free best-of-N: sample N candidate plans per replan (vision "
                   "amortized), pick by a FREE signal (no trained verifier). 0=off.")
@click.option('--bon_signal', default='vote', type=click.Choice(['vote', 'medoid', 'value']),
              help="ranking signal: 'vote'=mode-seeking KDE density, 'medoid'=min sum dist, "
                   "'value'=argmax ChunkQ critic (needs --chunk_q; Q-chunking QC analog)")
@click.option('--chunk_q', default=None, type=str,
              help="path to a ChunkQ critic .ckpt; attaches it so --bon_signal value ranks the "
                   "best-of-N candidates by learned Q(features, chunk) instead of consensus")
@click.option('--bon_prefix_k', default=0, type=int,
              help="coarse-to-fine BoN (idea #2): sample+select on the first bon_prefix_k "
                   "tokens (prefix-decoded to full chunk), then AR-refine the winner's tail "
                   "to use_k_tokens. 0=off (flat BoN over full budget).")
@click.option('--bon_first_temp', default=0.0, type=float,
              help="mode-injection BoN (idea #3): sample the FIRST token at this temperature "
                   "(inject mode diversity), continue the tail at base temperature. "
                   "0=off (uniform temperature). OAT-unique: targets the mode token.")
@click.option('--n_parallel_envs', default=None, type=int,
              help="parallel sim envs during eval (n_test unchanged). For A/B validation.")
@click.option('--n_test', default=None, type=int,
              help="override number of eval episodes for the runner.")
@click.option('--test_start_seed', default=None, type=int,
              help="override env episode seed base (episode i uses test_start_seed + i).")
@click.option('--env_task_name', default=None, type=str,
              help="override MetaworldRunner task_name (e.g. mt4 or box-close). "
                   "Use mt4 for interleaved MT4 eval; single subtask for per-task paper eval.")
def eval_policy_sim(
    checkpoint: str,
    output_dir: str,
    num_exp: int = 1,
    device: str = 'cuda:0',
    # policy inference args
    temperature: Optional[float] = None,
    topk: Optional[int] = None,
    use_k_tokens: Optional[int] = None,
    token_predictor: Optional[str] = None,
    entropy_threshold: Optional[float] = None,
    agnostic_mix: Optional[str] = None,
    n_action_steps: Optional[int] = None,
    adaptive_r: Optional[str] = None,
    r_coarse_k: int = 4,
    r_min: int = 8,
    r_max: int = 32,
    r_threshold: float = 0.5,
    bon_free: int = 0,
    bon_signal: str = 'vote',
    chunk_q: Optional[str] = None,
    bon_prefix_k: int = 0,
    bon_first_temp: float = 0.0,
    n_parallel_envs: Optional[int] = None,
    n_test: Optional[int] = None,
    test_start_seed: Optional[int] = None,
    env_task_name: Optional[str] = None,
):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
        os.system(f"rm -rf {output_dir}")
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # grab all checkpoints
    ckpts: List[str]    # file paths to checkpoints to evaluate
    if os.path.isdir(checkpoint):
        ckpts = [
            os.path.join(checkpoint, f) 
            for f in os.listdir(checkpoint) 
            if f.endswith('.ckpt') and f != 'latest.ckpt'
        ]
    else:
        ckpts = [checkpoint,]

    base_output_dir = output_dir
    for ckpt in ckpts:
        # format output dir
        if len(ckpts) > 1:
            ckpt_name = os.path.basename(ckpt).replace('.ckpt', '')
            output_dir = os.path.join(base_output_dir, ckpt_name)
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        else:
            output_dir = base_output_dir
        
        # load checkpoint
        policy, cfg = BasePolicy.from_checkpoint(ckpt, return_configuration=True)
        
        device = torch.device(device)
        policy.to(device)
        policy.eval()

        # optionally attach a learned token-count predictor for adaptive generation
        if token_predictor is not None:
            predictor = TokenCountPredictor.from_checkpoint(token_predictor)
            policy.set_token_predictor(predictor)
            print(f"Attached token-count predictor from {token_predictor}")

        # optionally attach a ChunkQ critic for value-guided best-of-N (--bon_signal value)
        if chunk_q is not None:
            from oat.model.chunk_q import ChunkQ
            policy.set_chunk_q(ChunkQ.from_checkpoint(chunk_q))
            print(f"Attached ChunkQ critic from {chunk_q} (value-guided BoN)")

        # optionally attach an obs-agnostic budget mixture (gate baseline)
        if agnostic_mix is not None:
            k_probs = {}
            for part in agnostic_mix.split(','):
                k_str, p_str = part.split(':')
                k_probs[int(k_str)] = float(p_str)
            policy.set_agnostic_mix(k_probs)
            print(f"Attached obs-agnostic budget mixture: {k_probs}")

        # variable-R mode needs the wrapper action_space to span the full r_max chunk
        # (each env then NaN-pads down to its own R), so force n_action_steps = r_max
        if adaptive_r is not None:
            n_action_steps = r_max
            print(f"Variable-R mode '{adaptive_r}': r in [{r_min},{r_max}], "
                  f"coarse_k={r_coarse_k}, threshold={r_threshold}")

        # optionally override executed chunk length R (fixed-R sweep): the policy slices
        # action_pred[:, :n_action_steps], so set it on the policy too (not just the runner)
        if n_action_steps is not None:
            policy.n_action_steps = n_action_steps
            print(f"Override executed chunk length R = {n_action_steps}")

        # run eval
        print(f"Running evaluation on {ckpt}")
        runner_overrides = {}
        if n_action_steps is not None:
            runner_overrides['n_action_steps'] = n_action_steps   # sync wrapper action_space to R
        if n_parallel_envs is not None:
            runner_overrides['n_parallel_envs'] = n_parallel_envs
            print(f"Override n_parallel_envs = {n_parallel_envs}")
        if n_test is not None:
            runner_overrides['n_test'] = n_test
            print(f"Override n_test = {n_test}")
        if test_start_seed is not None:
            runner_overrides['test_start_seed'] = test_start_seed
            print(f"Override test_start_seed = {test_start_seed}")
        if env_task_name is not None:
            runner_overrides['task_name'] = env_task_name
            print(f"Override env_runner.task_name = {env_task_name}")
        env_runner: BaseRunner = hydra.utils.instantiate(
            cfg.task.policy.env_runner,
            output_dir=output_dir,
            **runner_overrides,
        )
        
        kwargs = {}
        if temperature is not None:
            kwargs['temperature'] = temperature
        if topk is not None:
            kwargs['topk'] = topk
        if use_k_tokens is not None:
            kwargs['use_k_tokens'] = use_k_tokens
        if entropy_threshold is not None:
            kwargs['entropy_threshold'] = entropy_threshold
        if adaptive_r is not None:
            kwargs['adaptive_r'] = adaptive_r
            kwargs['r_coarse_k'] = r_coarse_k
            kwargs['r_min'] = r_min
            kwargs['r_max'] = r_max
            kwargs['r_threshold'] = r_threshold
        if bon_free and bon_free > 1:
            kwargs['bon_free'] = bon_free
            kwargs['bon_signal'] = bon_signal
            kwargs['bon_prefix_k'] = bon_prefix_k
            kwargs['bon_first_temp'] = bon_first_temp
            mode = f"coarse-to-fine prefix_k={bon_prefix_k}" if 0 < bon_prefix_k else "flat"
            if bon_first_temp > 0:
                mode += f", first_token_temp={bon_first_temp}"
            print(f"verifier-free BoN: N={bon_free}, signal={bon_signal}, {mode}")
        runner_log = env_runner.run(
            policy,
            **kwargs
        )
        
        # Store all runs for computing statistics
        all_runs = []
        for key, value in runner_log.items():
            if isinstance(value, wandb.sdk.data_types.video.Video):
                runner_log[key] = [value]
        all_runs.append({k: v for k, v in runner_log.items() if not isinstance(v, list)})
        print(f"Exp 1: success rate = {runner_log['mean_success_rate']}, mean tokens used = {runner_log.get('mean_tokens_used', 'N/A')}, mean R = {runner_log.get('mean_r_exec', 'N/A')}")

        for i in range(num_exp - 1):
            this_log = env_runner.run(policy, **kwargs)
            print(f"Exp {i + 2}: success rate = {this_log['mean_success_rate']}, mean tokens used = {this_log.get('mean_tokens_used', 'N/A')}")
            all_runs.append({k: v for k, v in this_log.items() if not isinstance(v, list)})
            # merge logs
            for key, value in this_log.items():
                assert key in runner_log
                if isinstance(value, wandb.sdk.data_types.video.Video):
                    runner_log[key].append(value)
                else:
                    runner_log[key] += value
        
        # Compute mean and std for all numeric metrics
        numeric_keys = [k for k in all_runs[0].keys()]
        mean_log = {}
        std_log = {}
        
        for key in numeric_keys:
            values = [run[key] for run in all_runs]
            mean_log[key] = np.mean(values)
            if num_exp > 1:
                std_log[key] = np.std(values, ddof=1)  # sample std
        
        env_runner.close()
        
        # dump log to json
        json_log = dict()
        json_log['checkpoint'] = ckpt
        json_log['num_exp'] = num_exp
        
        # Add mean values
        for key, value in mean_log.items():
            json_log[f'{key}_mean'] = float(value)
        
        # Add standard deviation & error values if multiple experiments
        if num_exp > 1:
            for key, value in std_log.items():
                json_log[f'{key}_std'] = float(value)
                json_log[f'{key}_stderr'] = float(value / np.sqrt(num_exp))
        
        # Add video paths
        for key, value in runner_log.items():
            if isinstance(value, list):
                for i, video in enumerate(value):
                    assert isinstance(video, wandb.sdk.data_types.video.Video)
                    json_log[f'{key}_{i}'] = video._path
        
        out_path = os.path.join(output_dir, 'eval_log.json')
        json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)


if __name__ == '__main__':
    eval_policy_sim()