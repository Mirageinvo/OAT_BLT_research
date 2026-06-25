"""
On-policy signed policy-gradient fine-tuning of the OAT AR head (the CRAFT-spirit ceiling-breaker:
unlike AWR's exp(adv/beta)>=0 weight which can only push actions UP, the SIGNED gradient pushes
GOOD chunks up AND BAD chunks DOWN -> can EXCEED the imitation ceiling).

  loss_i = adv_i * sum_t w_t * CE(logits_{i,t}, tokens_{i,t})  +  beta_kl * KL(pi || pi_ref)
         = -adv_i * logpi(tokens_i) + KL   (REINFORCE-with-baseline)
  adv_i  = succ_i - baseline(s_i)          (baseline: constant mean OR a ChunkQ V(s) critic),
           then standardized (mean0/std1) for stable step size.

CRITICAL — data MUST be ON-POLICY: collect with collect_awr_dataset.py using **--bon_n 0** so the
executed action is sampled from pi itself (NOT vote-selected), else logpi(action) is off the
policy's own distribution and the signed update is biased. Multi-epoch on a fixed batch drifts
off-policy -> keep --epochs LOW (1-3) and ITERATE (collect -> update -> recollect). KL-to-frozen-
ref + standardized advantage keep the step safe. Vision encoder & tokenizer stay FROZEN.

Run (one PG iteration):
  cd oat && MUJOCO_GL=egl uv run python scripts/collect_awr_dataset.py \
      -c my_models/policy_ep-0250_sr-0.596.ckpt -o my_datasets/pg_iter0.npz \
      --n_chunks 20000 --n_tasks 10 --bon_n 0 --n_workers 6
  uv run python scripts/train_pg.py -i my_datasets/pg_iter0.npz \
      -c my_models/policy_ep-0250_sr-0.596.ckpt -o my_models/policy_pg0.ckpt \
      --epochs 2 --beta_kl 0.1 [--critic my_models/chunk_q.ckpt]
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
@click.option('-i', '--input', 'inp', required=True, help='ON-POLICY dataset .npz (collect --bon_n 0)')
@click.option('-c', '--checkpoint', required=True, help='policy ckpt to fine-tune (= the collection policy)')
@click.option('-o', '--output', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('--beta_kl', default=0.1, type=float, help='KL-to-reference weight (higher than AWR: '
              'signed PG drifts faster, anchor harder)')
@click.option('--epochs', default=2, type=int, help='KEEP LOW (1-3): fixed batch drifts off-policy')
@click.option('--lr', default=5e-5, type=float, help='smaller than AWR (signed step is sharper)')
@click.option('--batch_size', default=256, type=int)
@click.option('--adv_clip', default=3.0, type=float, help='clip standardized advantage to [-c, c]')
@click.option('--ordering', default='uniform', type=click.Choice(['uniform', 'early', 'late']))
@click.option('--critic', default=None, type=str, help='ChunkQ ckpt for a V(s) baseline (else constant mean)')
def main(inp, checkpoint, output, device, beta_kl, epochs, lr, batch_size, adv_clip, ordering, critic):
    device = torch.device(device)

    # --- load workspace + policy ---
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    WorkspaceCls = hydra.utils.get_class(cfg._target_)
    workspace = WorkspaceCls(cfg, output_dir=None, lazy_instantiation=False)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.model
    policy.to(device)
    model = policy.model
    ref_model = copy.deepcopy(model).eval().requires_grad_(False)   # frozen ref for KL
    bos_id = policy.bos_id
    K = policy.max_seq_len

    # --- data ---
    d = np.load(inp, allow_pickle=True)
    feats = torch.from_numpy(d['features']).float()
    toks = torch.from_numpy(d['tokens']).long()
    succ = torch.from_numpy(d['success']).float()
    assert toks.shape[1] == K, f"token width {toks.shape[1]} != max_seq_len {K}"
    ep_sr = float(succ.mean())

    # --- advantage = succ - baseline, then standardized ---
    if critic is not None:
        from oat.model.chunk_q import ChunkQ
        cq = ChunkQ.from_checkpoint(critic).to(device).eval()
        norm = policy.action_tokenizer.normalizer['action']
        vlist = []
        with torch.inference_mode():
            for i in range(0, len(toks), 512):
                tk = toks[i:i + 512].to(device)
                a = policy.action_tokenizer.detokenize(tk, eval_keep_k=[K] * tk.shape[0])
                vlist.append(cq.score(feats[i:i + 512].to(device), norm.normalize(a[:, :cq.horizon])).cpu())
        baseline = torch.cat(vlist)
        bl_desc = f"V(s) mean {baseline.mean():.3f}"
    else:
        baseline = succ.mean()
        bl_desc = f"const {float(baseline):.3f}"
    adv = succ - baseline
    adv = (adv - adv.mean()) / (adv.std() + 1e-6)                   # standardize -> stable step
    adv = adv.clamp(-adv_clip, adv_clip)
    print(f"N={len(feats)}  on-policy SR={ep_sr:.3f}  baseline={bl_desc}  "
          f"adv(std)[min/mean/max]={adv.min():.2f}/{adv.mean():.2f}/{adv.max():.2f}  "
          f"frac_neg={float((adv < 0).float().mean()):.2f}  ordering={ordering}")

    # per-token ordering credit
    pos = torch.arange(1, K + 1, dtype=torch.float32)
    wt = (1.0 / pos) if ordering == 'early' else (pos if ordering == 'late' else torch.ones(K))
    wt = (wt / wt.mean()).to(device)
    bos_t = torch.full((1, 1), bos_id, dtype=torch.long)

    ds = torch.utils.data.TensorDataset(feats, toks, adv)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for ep in range(epochs):
        tot_pg, tot_kl, nb = 0.0, 0.0, 0
        for fb, tb, ab in tqdm.tqdm(loader, desc=f'epoch {ep+1}/{epochs}'):
            fb, tb, ab = fb.to(device), tb.to(device), ab.to(device)
            B = fb.shape[0]
            inp_tok = torch.cat([bos_t.expand(B, 1).to(device), tb[:, :-1]], dim=1)
            logits = model(inp_tok, cond=fb)
            V = logits.size(-1)
            ce = F.cross_entropy(logits.reshape(-1, V), tb.reshape(-1),
                                 reduction='none').reshape(B, K)
            ce_w = (ce * wt[None, :]).sum(1)                        # [B]
            pg_loss = (ab * ce_w).mean()                           # SIGNED: adv<0 pushes logpi DOWN
            with torch.no_grad():
                ref_logits = ref_model(inp_tok, cond=fb)
            kl = (F.softmax(logits, -1) *
                  (F.log_softmax(logits, -1) - F.log_softmax(ref_logits, -1))).sum(-1).mean()
            loss = pg_loss + beta_kl * kl
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_pg += pg_loss.item(); tot_kl += kl.item(); nb += 1
        print(f"  epoch {ep+1}: pg_loss={tot_pg/nb:+.4f}  kl={tot_kl/nb:.4f}")

    model.eval()
    if getattr(workspace, 'ema_model', None) is not None:
        try:
            workspace.ema_model.model.load_state_dict(model.state_dict())
        except Exception as e:
            print(f"warn: could not sync ema_model: {e}")
    pathlib.Path(output).parent.mkdir(parents=True, exist_ok=True)
    workspace.save_checkpoint(path=output, use_thread=False)
    print(f"saved PG checkpoint -> {output}")


if __name__ == '__main__':
    main()
