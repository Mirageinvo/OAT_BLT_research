"""
Offline calibration of the PACE prominence threshold (NO sim). For each candidate prominence,
compute the speed-valley execution horizon r_exec on stored features (generate -> decode ->
speed profile -> first prominent low-speed valley) and report the resulting mean R. Pick the
threshold giving a target mean R (e.g. ~16) so the variable-R sim can be matched-cost vs random.

Mirror of the 'pace' branch in OATPolicy.predict_action_variable_r.

Run:
  cd oat && uv run python scripts/calib_pace_threshold.py \
      -c my_models/policy_ep-0250_sr-0.596.ckpt -i my_datasets/awr_bon.npz
"""
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import click
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import find_peaks

from oat.policy.base_policy import BasePolicy
from oat.policy.oatpolicy import OATPolicy


@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-i', '--input', 'inp', default='my_datasets/awr_bon.npz', help='features source (.npz)')
@click.option('-d', '--device', default='cuda:0')
@click.option('--r_min', default=8, type=int)
@click.option('--r_max', default=32, type=int)
@click.option('--n', default=3000, type=int, help='num feature rows to use')
@click.option('--thresholds', default='0.03,0.05,0.08,0.1,0.12,0.15,0.2,0.3',
              help='comma-separated prominence thresholds to sweep')
@click.option('--raw', is_flag=True, default=False,
              help="pace_raw variant: speed = ||raw EE-translation[:3]|| (no normalizer)")
def main(checkpoint, inp, device, r_min, r_max, n, thresholds, raw):
    device = torch.device(device)
    pol = BasePolicy.from_checkpoint(checkpoint)
    pol.to(device).eval()
    assert isinstance(pol, OATPolicy)
    K = pol.max_seq_len
    norm = pol.action_tokenizer.normalizer['action']

    feats = torch.from_numpy(np.load(inp, allow_pickle=True)['features'][:n]).float().to(device)
    bs, speed_chunks = 256, []
    with torch.inference_mode():
        for i in range(0, feats.shape[0], bs):
            fb = feats[i:i + bs]
            B = fb.shape[0]
            bos = torch.full((B, 1), pol.bos_id, dtype=torch.long, device=device)
            tk = pol.model.generate(bos, cond=fb, max_new_tokens=K,
                                    temperature=pol.temperature, top_k=pol.topk)[:, 1:]
            a = pol.action_tokenizer.detokenize(tk, eval_keep_k=[K] * B)        # [B,H,D] raw
            sp = (a[:, :r_max, :3].norm(dim=-1) if raw                          # [B,r_max] speed
                  else norm.normalize(a)[:, :r_max, :6].norm(dim=-1))
            speed_chunks.append(sp.cpu())
    speed = torch.cat(speed_chunks, 0)                                         # [N,r_max]
    kernel = torch.ones(1, 1, 3) / 3.0                                         # smooth (window 3)
    sp_np = F.conv1d(F.pad(speed[:, None, :], (1, 1), mode='replicate'), kernel)[:, 0, :].numpy()

    print(f"N={sp_np.shape[0]}  r_max={r_max}  r_min={r_min}  (target mean R ~ 16 for matched-cost)")
    for thr in [float(x) for x in thresholds.split(',')]:
        rs = np.empty(sp_np.shape[0], dtype=np.int64)
        for b in range(sp_np.shape[0]):
            v, _ = find_peaks(-sp_np[b], prominence=thr)
            r = (int(v[0]) + 1) if len(v) else r_max
            rs[b] = min(max(r, r_min), r_max)
        print(f"  prominence={thr:<5}  meanR={rs.mean():5.2f}  std={rs.std():4.2f}  "
              f"%@rmin={np.mean(rs <= r_min) * 100:4.0f}  %@rmax={np.mean(rs >= r_max) * 100:4.0f}")


if __name__ == '__main__':
    main()
