"""
Collect an AWR / ReST fine-tuning dataset by rolling out the policy in LIBERO and labeling
each executed chunk with its EPISODE success (cheap reward — no counterfactual sim).

Per replan we log (features = obs_encoder(obs), executed action tokens); at episode end we
broadcast the binary episode success to every chunk of that episode. AWR then does weighted
SFT: weight = exp((success - baseline)/beta), so successful-episode chunks are up-weighted.

  --bon_n N  : roll out WITH verifier-free BoN selection (vote) and log the SELECTED tokens
               -> distills the +0.11 BoN policy into the AR head (BoN-distillation).
               default 0/1 = base policy (plain ReST on the base policy's own successes).

Features (not raw obs) are stored: obs_encoder is frozen, so features are fixed, and obs
(128x128 images) are huge. Output .npz: features [N,To,d], tokens [N,K], success [N],
ep_id [N], step [N], task [N(object)].

Run (sim machine):
  cd oat && MUJOCO_GL=egl uv run python scripts/collect_awr_dataset.py \
      -c my_models/policy_ep-0250_sr-0.596.ckpt -o my_datasets/awr_bon.npz \
      --n_chunks 20000 --bon_n 8 --n_tasks 10
"""
import sys, os, pathlib
ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
if __name__ == "__main__":
    os.chdir(ROOT_DIR)

from collections import deque

import click
import numpy as np
import torch
import tqdm

from oat.policy.base_policy import BasePolicy
from oat.policy.oatpolicy import OATPolicy
from oat.env.libero.env import LiberoEnv
from oat.env.libero.factory import get_subtasks
from scripts.branch_value_k import build_obs, decode_k, env_kwargs_from_cfg


@torch.inference_mode()
def gen_action_tokens(policy, features, bon_n, n_act):
    """Sample executed action tokens at this obs. bon_n<=1: base policy. bon_n>1: verifier-free
    BoN (sample N on shared features, vote-select over the executed prefix), return the
    SELECTED token sequence. Returns tokens [1, K]."""
    K = policy.max_seq_len
    bos = lambda n: torch.full((n, 1), policy.bos_id, dtype=torch.long, device=policy.device)
    if bon_n is None or bon_n <= 1:
        return policy.model.generate(bos(1), cond=features, max_new_tokens=K,
                                     temperature=policy.temperature, top_k=policy.topk)[:, 1:]
    feat_rep = features.repeat_interleave(bon_n, dim=0)                      # [N, To, d]
    toks = policy.model.generate(bos(bon_n), cond=feat_rep, max_new_tokens=K,
                                 temperature=policy.temperature, top_k=policy.topk)[:, 1:]  # [N,K]
    cand = policy.action_tokenizer.detokenize(toks, eval_keep_k=[K] * bon_n)  # [N,H,D]
    R = min(n_act, cand.shape[1])
    best = policy._bon_select(cand, R, 'vote')
    return toks[best:best + 1]                                               # [1, K]


@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('--n_chunks', default=20000, type=int, help='target number of logged chunks')
@click.option('--n_tasks', default=10, type=int)
@click.option('--bon_n', default=0, type=int, help='0/1=base policy; >1=BoN-distillation (log selected)')
@click.option('--temperature', default=None, type=float)
@click.option('--topk', default=None, type=int)
@click.option('--seed', default=0, type=int)
def main(checkpoint, output, device, n_chunks, n_tasks, bon_n, temperature, topk, seed):
    device = torch.device(device)
    policy, cfg = BasePolicy.from_checkpoint(checkpoint, return_configuration=True)
    assert isinstance(policy, OATPolicy)
    policy.to(device).eval()
    if temperature is not None:
        policy.temperature = temperature
    if topk is not None:
        policy.topk = topk
    dtype = policy.dtype
    ports = policy.get_observation_ports()
    n_obs = policy.n_obs_steps
    n_act = policy.n_action_steps
    ekw = env_kwargs_from_cfg(cfg)
    try:
        suite = cfg.task.policy.env_runner.task_name
    except Exception:
        suite = 'libero10'
    tasks = get_subtasks(suite)[:n_tasks]
    mode = f"BoN-distill(N={bon_n})" if bon_n > 1 else "base ReST"
    print(f"mode={mode} | n_chunks={n_chunks} | tasks={len(tasks)} | n_act(R)={n_act}")

    feats, toks, succ, ep_ids, steps, task_ids = [], [], [], [], [], []
    ep_id = 0
    n_ep_done = 0
    pbar = tqdm.tqdm(total=n_chunks, desc='chunks')
    for task in tasks:
        if len(feats) >= n_chunks:
            break
        env = LiberoEnv(task_name=task, seed=seed, **ekw)
        try:
            env.env.env.ignore_done = True
        except Exception:
            pass
        try:
            while len(feats) < n_chunks:
                obs, _ = env.reset()
                obs_deque = deque([obs], maxlen=n_obs + 1)
                ep_start = len(feats)
                ep_success = False
                while not env.done and env.cur_step < env.max_episode_steps and len(feats) < n_chunks:
                    obs_dict = build_obs(obs_deque, n_obs, ports, device, dtype)
                    with torch.inference_mode():
                        features = policy.obs_encoder(obs_dict)             # [1, To, d]
                    tk = gen_action_tokens(policy, features, bon_n, n_act)  # [1, K]
                    a = decode_k(policy, tk, policy.max_seq_len)            # [H, D]
                    feats.append(features[0].detach().cpu().numpy())
                    toks.append(tk[0].detach().cpu().numpy())
                    ep_ids.append(ep_id); steps.append(int(env.cur_step)); task_ids.append(task)
                    succ.append(0.0)                                        # placeholder, filled below
                    pbar.update(1)
                    for t in range(n_act):
                        if env.done:
                            break
                        obs, r, done, _, _ = env.step(a[t])
                        obs_deque.append(obs)
                        ep_success = ep_success or (r >= 1)
                for i in range(ep_start, len(succ)):
                    succ[i] = float(ep_success)                            # broadcast episode label
                ep_id += 1
                n_ep_done += 1
        finally:
            env.close()
    pbar.close()

    feats = np.asarray(feats, dtype=np.float32)
    toks = np.asarray(toks, dtype=np.int64)
    succ = np.asarray(succ, dtype=np.float32)
    pathlib.Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, features=feats, tokens=toks, success=succ,
             ep_id=np.asarray(ep_ids), step=np.asarray(steps),
             task=np.asarray(task_ids, dtype=object))
    print(f"\nsaved {len(feats)} chunks from {n_ep_done} episodes -> {output}")
    print(f"  features {feats.shape}  tokens {toks.shape}")
    print(f"  episode success rate (chunk-weighted) = {succ.mean():.3f}  "
          f"(baseline for AWR advantage)")
    ep_arr = np.asarray(ep_ids)
    ep_sr = [succ[ep_arr == e][0] for e in np.unique(ep_arr) if (ep_arr == e).any()]
    if ep_sr:
        print(f"  per-episode SR = {np.mean(ep_sr):.3f}  over {len(ep_sr)} episodes")


if __name__ == '__main__':
    main()
