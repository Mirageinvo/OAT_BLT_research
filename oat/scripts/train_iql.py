"""
IQL (Implicit Q-Learning) fine-tuning of the OAT AR head — the stable, conservative offline-RL
upgrade of our failed chunk-Q (0.476, OOD) and signed-PG (0.182, collapse). Chunk-level MDP:
each replan = one transition (s=features, a=executed chunk, r=0 except terminal=episode success,
s'=next replan's features). Two things IQL gives us:
  1. conservative critic WITHOUT OOD queries (expectile-V never maxes over off-data actions) ->
     no outlier-picking (chunk-Q) and no collapse (signed-PG);
  2. chunk-level TD credit (A = r + gamma*V(s') - V(s)) -> shorter horizon than the episode-
     success-broadcast baseline our AWR used -> TESTS "is TD credit cleaner than broadcast?".

Pipeline (offline, NO sim; reuses collect_awr_dataset .npz + ChunkQ/ChunkV):
  A) train IQL critic: V <- expectile_tau(Q_target(s,a)),  Q <- r + gamma*(1-done)*V(s') (MSE),
     Q_target Polyak. B) extract policy by AWR-weighting the AR head with adv = Q(s,a) - V(s):
     loss = exp(adv/beta)*CE + beta_kl*KL(pi||pi_ref).  Vision+tokenizer FROZEN.

NB IQL is AWR-family -> imitation-bounded; expect ~0.68 UNLESS TD credit beats broadcast. It is
the STABLE PRIOR for the online-RL ceiling-break (residual on top), not a ceiling-break itself.

Run:
  cd oat && uv run python scripts/train_iql.py -i my_datasets/awr_bon16.npz \
      -c my_models/policy_ep-0250_sr-0.596.ckpt -o my_models/policy_iql.ckpt \
      --gamma 0.99 --expectile 0.7 --beta 3.0 --critic_steps 20000 --epochs 30
Eval: eval_policy_sim.py -c policy_iql.ckpt --entropy_threshold 0 --use_k_tokens 8  (vs AWR 0.684)
"""
import sys, os, pathlib
ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
if __name__ == "__main__":
    os.chdir(ROOT_DIR)

import copy

import click
import dill
import hydra
import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from oat.model.chunk_q import ChunkQ, ChunkV


def expectile_loss(diff: torch.Tensor, tau: float) -> torch.Tensor:
    """Asymmetric L2: weight tau on positive residuals, (1-tau) on negative -> V regresses toward
    the tau-expectile of Q (tau>0.5 = optimistic ~ soft-max over in-support actions)."""
    w = torch.where(diff > 0, tau, 1.0 - tau)
    return (w * diff.pow(2)).mean()


@torch.inference_mode()
def detok_chunks(policy, tokens, horizon, device, bs=512):
    """tokens [N,K] -> normalized chunk [N, horizon, D] (normalizer space)."""
    K = policy.max_seq_len
    norm = policy.action_tokenizer.normalizer['action']
    out = []
    for i in range(0, tokens.shape[0], bs):
        tk = torch.from_numpy(tokens[i:i + bs]).long().to(device)
        a = policy.action_tokenizer.detokenize(tk, eval_keep_k=[K] * tk.shape[0])
        out.append(norm.normalize(a[:, :horizon]).cpu())
    return torch.cat(out, 0)


@click.command()
@click.option('-i', '--input', 'inp', required=True, help='collect_awr_dataset .npz')
@click.option('-c', '--checkpoint', required=True, help='base policy ckpt to fine-tune')
@click.option('-o', '--output', required=True)
@click.option('-d', '--device', default='cuda:0')
# --- IQL critic ---
@click.option('--gamma', default=0.99, type=float, help='per-chunk discount')
@click.option('--expectile', default=0.7, type=float, help='IQL tau (V toward upper expectile of Q)')
@click.option('--critic_steps', default=20000, type=int)
@click.option('--critic_lr', default=3e-4, type=float)
@click.option('--critic_bs', default=512, type=int)
@click.option('--polyak', default=0.005, type=float, help='Q_target Polyak rate')
# --- AWR policy extraction ---
@click.option('--beta', default=3.0, type=float, help='AWR temperature on adv=Q-V (adv in ~[-1,1])')
@click.option('--w_max', default=20.0, type=float)
@click.option('--beta_kl', default=0.05, type=float)
@click.option('--epochs', default=30, type=int)
@click.option('--lr', default=1e-4, type=float)
@click.option('--batch_size', default=256, type=int)
@click.option('--save_critic', default=None, type=str, help='optional path to dump the Q/V critic')
def main(inp, checkpoint, output, device, gamma, expectile, critic_steps, critic_lr, critic_bs,
         polyak, beta, w_max, beta_kl, epochs, lr, batch_size, save_critic):
    device = torch.device(device)

    # --- load workspace + policy (AR head + detokenize + save) ---
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    WorkspaceCls = hydra.utils.get_class(cfg._target_)
    workspace = WorkspaceCls(cfg, output_dir=None, lazy_instantiation=False)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.model
    policy.to(device).eval()
    model = policy.model                      # AR head (only thing trained in phase B)
    bos_id = policy.bos_id
    K = policy.max_seq_len
    horizon = policy.n_action_steps

    # --- data -> chunk-level transitions (sort by episode, then step) ---
    d = np.load(inp, allow_pickle=True)
    feats = d['features'].astype(np.float32)          # [N, To, dd]
    toks = d['tokens'].astype(np.int64)
    succ = d['success'].astype(np.float32)
    ep = d['ep_id']; step = d['step']
    order = np.lexsort((step, ep))                    # sort by (ep_id, step)
    feats, toks, succ, ep = feats[order], toks[order], succ[order], ep[order]
    N, To, dd = feats.shape

    # terminal = last chunk of its episode; reward at terminal = episode success, else 0
    done = np.zeros(N, dtype=np.float32)
    done[-1] = 1.0
    done[:-1] = (ep[1:] != ep[:-1]).astype(np.float32)
    next_idx = np.minimum(np.arange(N) + 1, N - 1)    # i+1 (masked out when done)
    reward = np.where(done > 0.5, succ, 0.0).astype(np.float32)
    print(f"N={N} transitions | episodes={len(np.unique(ep))} | terminal SR={succ[done>0.5].mean():.3f} "
          f"| gamma={gamma} tau={expectile}")

    feats_t = torch.from_numpy(feats)
    chunks_t = detok_chunks(policy, toks, horizon, device)     # [N, horizon, D] (cpu)
    D = chunks_t.shape[-1]
    reward_t = torch.from_numpy(reward)
    done_t = torch.from_numpy(done)
    next_t = torch.from_numpy(next_idx).long()

    fm = feats_t.reshape(N, -1).mean(0)
    fs = feats_t.reshape(N, -1).std(0)

    # ================= Phase A: IQL critic =================
    Q = ChunkQ(in_dim=dd, n_obs_steps=To, horizon=horizon, action_dim=D).to(device)
    Qtarg = copy.deepcopy(Q).eval().requires_grad_(False)
    V = ChunkV(in_dim=dd, n_obs_steps=To).to(device)
    for m in (Q, V):
        m.set_feature_stats(fm, fs)
    Qtarg.set_feature_stats(fm, fs)
    optQ = torch.optim.Adam(Q.parameters(), lr=critic_lr)
    optV = torch.optim.Adam(V.parameters(), lr=critic_lr)

    def batch(idx):
        return (feats_t[idx].to(device), chunks_t[idx].to(device), reward_t[idx].to(device),
                done_t[idx].to(device), feats_t[next_t[idx]].to(device))

    Q.train(); V.train()
    for it in tqdm.tqdm(range(critic_steps), desc='IQL critic'):
        idx = torch.randint(0, N, (critic_bs,))
        s, a, r, dn, s2 = batch(idx)
        # V update: expectile regression toward Q_target(s,a)
        with torch.no_grad():
            q_targ = Qtarg(s, a)
        v = V(s)
        v_loss = expectile_loss(q_targ - v, expectile)
        optV.zero_grad(); v_loss.backward(); optV.step()
        # Q update: TD toward r + gamma*(1-done)*V(s')
        with torch.no_grad():
            td = r + gamma * (1.0 - dn) * V(s2)
        q = Q(s, a)
        q_loss = F.mse_loss(q, td)
        optQ.zero_grad(); q_loss.backward(); optQ.step()
        # Polyak target
        with torch.no_grad():
            for p, pt in zip(Q.parameters(), Qtarg.parameters()):
                pt.mul_(1 - polyak).add_(polyak * p)
        if (it + 1) % max(1, critic_steps // 10) == 0:
            with torch.no_grad():
                adv = (Q(s, a) - V(s))
            tqdm.tqdm.write(f"  step {it+1}: v_loss={v_loss.item():.4f} q_loss={q_loss.item():.4f} "
                            f"| V[min/mean/max]={v.min():.2f}/{v.mean():.2f}/{v.max():.2f} "
                            f"adv[min/mean/max]={adv.min():.2f}/{adv.mean():.2f}/{adv.max():.2f}")

    # advantage over the WHOLE dataset (original order for the AR-head SFT below)
    Q.eval(); V.eval()
    with torch.no_grad():
        adv = torch.empty(N)
        for i in range(0, N, 4096):
            s = feats_t[i:i+4096].to(device); a = chunks_t[i:i+4096].to(device)
            adv[i:i+4096] = (Q(s, a) - V(s)).cpu()
    weight = torch.exp(adv / beta).clamp(max=w_max)
    weight = weight / weight.mean()
    frac_neg = float((adv < 0).float().mean())
    print(f"[extract] adv[min/mean/max]={adv.min():.2f}/{adv.mean():.2f}/{adv.max():.2f} "
          f"frac_neg={frac_neg:.2f} | weight[min/mean/max]={weight.min():.2f}/{weight.mean():.2f}/{weight.max():.2f}")
    if save_critic:
        torch.save({'q': {'config': dict(in_dim=dd, n_obs_steps=To, horizon=horizon, action_dim=D,
                                          hidden_dims=(256, 256), dropout=0.1), 'state': Q.state_dict()},
                    'v': {'config': dict(in_dim=dd, n_obs_steps=To, hidden_dims=(256, 256), dropout=0.1),
                          'state': V.state_dict()}}, save_critic)
        print(f"saved critic -> {save_critic}")

    # ================= Phase B: AWR policy extraction (AR head) =================
    ref_model = copy.deepcopy(model).eval().requires_grad_(False)
    wt = torch.ones(K, device=device)                          # uniform per-token credit
    bos_t = torch.full((1, 1), bos_id, dtype=torch.long)
    # data in ORIGINAL order (weights already aligned to sorted order -> reorder back)
    inv = np.empty(N, dtype=np.int64); inv[order] = np.arange(N)   # map sorted->original
    feats_orig = torch.from_numpy(d['features'].astype(np.float32))
    toks_orig = torch.from_numpy(d['tokens'].astype(np.int64))
    weight_orig = weight[torch.from_numpy(inv)]                 # align weights to original rows

    ds = torch.utils.data.TensorDataset(feats_orig, toks_orig, weight_orig)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for e in range(epochs):
        tot, tkl, nb = 0.0, 0.0, 0
        for fb, tb, wb in tqdm.tqdm(loader, desc=f'AWR-extract {e+1}/{epochs}'):
            fb, tb, wb = fb.to(device), tb.to(device), wb.to(device)
            B = fb.shape[0]
            inp_tok = torch.cat([bos_t.expand(B, 1).to(device), tb[:, :-1]], dim=1)
            logits = model(inp_tok, cond=fb)
            Vv = logits.size(-1)
            ce = F.cross_entropy(logits.reshape(-1, Vv), tb.reshape(-1), reduction='none').reshape(B, K)
            awr_loss = (wb * (ce * wt[None, :]).sum(1)).mean()
            with torch.no_grad():
                ref_logits = ref_model(inp_tok, cond=fb)
            kl = (F.softmax(logits, -1) * (F.log_softmax(logits, -1) - F.log_softmax(ref_logits, -1))).sum(-1).mean()
            loss = awr_loss + beta_kl * kl
            opt.zero_grad(); loss.backward(); opt.step()
            tot += awr_loss.item(); tkl += kl.item(); nb += 1
        print(f"  epoch {e+1}: awr_loss={tot/nb:.4f} kl={tkl/nb:.4f}")

    model.eval()
    if getattr(workspace, 'ema_model', None) is not None:
        try:
            workspace.ema_model.model.load_state_dict(model.state_dict())
        except Exception as ex:
            print(f"warn: could not sync ema_model: {ex}")
    pathlib.Path(output).parent.mkdir(parents=True, exist_ok=True)
    workspace.save_checkpoint(path=output, use_thread=False)
    print(f"saved IQL-extracted policy -> {output}")


if __name__ == '__main__':
    main()
