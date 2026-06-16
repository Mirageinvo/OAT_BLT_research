"""
AWR / ReST fine-tuning of the OAT AR head on a collected dataset (collect_awr_dataset.py).

Advantage-weighted SFT: weight_i = clip(exp((success_i - baseline)/beta)), loss = weight_i *
sum_t w_t * CE(logits_{i,t}, tokens_{i,t}) + beta_kl * KL(pi || pi_ref). Vision encoder &
tokenizer are FROZEN (we trained on stored features); only the AR head (policy.model) is
updated. KL to a frozen reference copy = anti-collapse / anti-drift (keeps diversity).

  w_t (ordering credit, the prefix-PROBE): 'uniform' | 'early' (1/t, weight the mode/first
  tokens — tests "decision is in early tokens") | 'late'. Run all three for the ablation.

Saves a checkpoint loadable by eval_policy_sim.py (writes both model & ema_model).

Run:
  cd oat && uv run python scripts/train_awr.py -i my_datasets/awr_bon.npz \
      -c my_models/policy_ep-0250_sr-0.596.ckpt -o my_models/policy_awr.ckpt \
      --beta 0.5 --beta_kl 0.05 --epochs 5 --ordering uniform
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


@click.command()
@click.option('-i', '--input', 'inp', required=True, help='awr dataset .npz')
@click.option('-c', '--checkpoint', required=True, help='base policy ckpt to fine-tune')
@click.option('-o', '--output', required=True, help='output ckpt path')
@click.option('-d', '--device', default='cuda:0')
@click.option('--beta', default=0.5, type=float, help='AWR temperature (smaller = sharper weighting)')
@click.option('--beta_kl', default=0.05, type=float, help='KL-to-reference weight (anti-collapse)')
@click.option('--w_max', default=20.0, type=float, help='clip on AWR weight')
@click.option('--epochs', default=5, type=int)
@click.option('--lr', default=1e-4, type=float)
@click.option('--batch_size', default=256, type=int)
@click.option('--ordering', default='uniform', type=click.Choice(['uniform', 'early', 'late']),
              help='per-token credit w_t (prefix-probe): early=1/t weights the mode tokens')
def main(inp, checkpoint, output, device, beta, beta_kl, w_max, epochs, lr, batch_size, ordering):
    device = torch.device(device)

    # --- load workspace + policy (so we can save_checkpoint back) ---
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    WorkspaceCls = hydra.utils.get_class(cfg._target_)
    workspace = WorkspaceCls(cfg, output_dir=None, lazy_instantiation=False)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.model
    policy.to(device)
    model = policy.model                                   # the AR head (only thing we train)
    ref_model = copy.deepcopy(model).eval().requires_grad_(False)  # frozen reference for KL
    bos_id = policy.bos_id
    K = policy.max_seq_len

    # --- data ---
    d = np.load(inp, allow_pickle=True)
    feats = torch.from_numpy(d['features']).float()        # [N, To, dim]
    toks = torch.from_numpy(d['tokens']).long()            # [N, K]
    succ = torch.from_numpy(d['success']).float()          # [N]
    assert toks.shape[1] == K, f"token width {toks.shape[1]} != max_seq_len {K}"
    baseline = succ.mean().item()
    adv = succ - baseline
    weight = torch.exp(adv / beta).clamp(max=w_max)        # AWR weight
    weight = weight / weight.mean()                        # normalize -> mean 1 (stable LR)
    print(f"N={len(feats)}  baseline(SR)={baseline:.3f}  weight[min/mean/max]="
          f"{weight.min():.2f}/{weight.mean():.2f}/{weight.max():.2f}  ordering={ordering}")

    # per-token ordering credit w_t
    pos = torch.arange(1, K + 1, dtype=torch.float32)
    if ordering == 'early':
        wt = 1.0 / pos
    elif ordering == 'late':
        wt = pos
    else:
        wt = torch.ones(K)
    wt = (wt / wt.mean()).to(device)                       # normalize -> mean 1
    bos_t = torch.full((1, 1), bos_id, dtype=torch.long)

    ds = torch.utils.data.TensorDataset(feats, toks, weight)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for ep in range(epochs):
        tot_awr, tot_kl, nb = 0.0, 0.0, 0
        for fb, tb, wb in tqdm.tqdm(loader, desc=f'epoch {ep+1}/{epochs}'):
            fb, tb, wb = fb.to(device), tb.to(device), wb.to(device)
            B = fb.shape[0]
            inp_tok = torch.cat([bos_t.expand(B, 1).to(device), tb[:, :-1]], dim=1)  # [B, K]
            logits = model(inp_tok, cond=fb)               # [B, K, V]
            V = logits.size(-1)
            ce = F.cross_entropy(logits.reshape(-1, V), tb.reshape(-1),
                                 reduction='none').reshape(B, K)            # [B, K]
            ce_w = (ce * wt[None, :]).sum(1)               # [B] position-weighted CE
            awr_loss = (wb * ce_w).mean()
            with torch.no_grad():
                ref_logits = ref_model(inp_tok, cond=fb)
            kl = (F.softmax(logits, -1) *
                  (F.log_softmax(logits, -1) - F.log_softmax(ref_logits, -1))).sum(-1).mean()
            loss = awr_loss + beta_kl * kl
            opt.zero_grad(); loss.backward(); opt.step()
            tot_awr += awr_loss.item(); tot_kl += kl.item(); nb += 1
        print(f"  epoch {ep+1}: awr_loss={tot_awr/nb:.4f}  kl={tot_kl/nb:.4f}")

    # write fine-tuned weights into BOTH model and ema_model (eval may load ema)
    model.eval()
    if getattr(workspace, 'ema_model', None) is not None:
        try:
            workspace.ema_model.model.load_state_dict(model.state_dict())
        except Exception as e:
            print(f"warn: could not sync ema_model: {e}")
    pathlib.Path(output).parent.mkdir(parents=True, exist_ok=True)
    workspace.save_checkpoint(path=output, use_thread=False)   # synchronous (script exits after)
    print(f"saved fine-tuned checkpoint -> {output}")


if __name__ == '__main__':
    main()
