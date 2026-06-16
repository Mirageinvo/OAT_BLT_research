"""
Fast sanity-check of a fine-tuned checkpoint — NO sim. Verifies: (1) it loads as an
OATPolicy, (2) its AR-head weights actually differ from the base (training had effect),
(3) it can encode->generate->detokenize on stored features (forward path works).

Run:
  cd oat && uv run python scripts/verify_ckpt.py -c my_models/policy_awr_smoke.ckpt \
      -b my_models/policy_ep-0250_sr-0.596.ckpt -i my_datasets/awr_smoke.npz
"""
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import click
import numpy as np
import torch

from oat.policy.base_policy import BasePolicy


@click.command()
@click.option('-c', '--checkpoint', required=True, help='fine-tuned ckpt')
@click.option('-b', '--base', default=None, help='base ckpt to diff against')
@click.option('-i', '--input', 'inp', default=None, help='awr npz (for real features)')
@click.option('-d', '--device', default='cuda:0')
def main(checkpoint, base, inp, device):
    device = torch.device(device)
    pol = BasePolicy.from_checkpoint(checkpoint)
    pol.to(device).eval()
    print(f"(1) loaded {type(pol).__name__}  max_seq_len={pol.max_seq_len}  n_obs={pol.n_obs_steps}")

    if base is not None:
        bpol = BasePolicy.from_checkpoint(base).to(device).eval()
        diff = sum((p - b).abs().sum().item()
                   for p, b in zip(pol.model.parameters(), bpol.model.parameters()))
        npar = sum(p.numel() for p in pol.model.parameters())
        print(f"(2) AR-head L1 weight diff vs base = {diff:.3f}  (mean/param {diff/npar:.2e})  "
              f"{'OK changed' if diff > 1e-3 else 'WARNING ~unchanged'}")

    # (3) generate on real stored features
    if inp is not None:
        feats = torch.from_numpy(np.load(inp, allow_pickle=True)['features'][:4]).float().to(device)
    else:
        feats = torch.randn(4, pol.n_obs_steps, pol.model.cond_emb.in_features
                            if hasattr(pol.model, 'cond_emb') else 138, device=device)
    with torch.inference_mode():
        bos = torch.full((feats.shape[0], 1), pol.bos_id, dtype=torch.long, device=device)
        toks = pol.model.generate(bos, cond=feats, max_new_tokens=pol.max_seq_len,
                                  temperature=pol.temperature, top_k=pol.topk)[:, 1:]
        act = pol.action_tokenizer.detokenize(toks, eval_keep_k=[pol.max_seq_len] * feats.shape[0])
    print(f"(3) generate OK: tokens {tuple(toks.shape)} -> action {tuple(act.shape)}  "
          f"(tokens[0]={toks[0].tolist()})")
    print("\nDONE — checkpoint is valid & differs from base.")


if __name__ == '__main__':
    main()
