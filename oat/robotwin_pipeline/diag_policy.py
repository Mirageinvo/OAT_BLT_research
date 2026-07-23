"""
Decisive offline check (NO sim): run the trained OAT policy on TRAINING obs from the Zarr and
see whether the predicted action depends on the observation.

  - actions VARY across different train obs  -> policy is fine (obs-dependent) -> the sim failure
    is an ADAPTER obs/action mismatch (eval obs != train obs). Fix the adapter.
  - actions are ~CONSTANT across train obs   -> mean-collapse -> UNDERFIT (or a broken normalizer).

Run on docker with the conda python:
  cd oat && source robotwin_pipeline/config.sh
  ${OATPY} robotwin_pipeline/diag_policy.py
"""
import os, numpy as np, torch, zarr

CKPT = os.environ["POLICY_CKPT"]
ZARR = os.environ["ZARR"]

from oat.policy.base_policy import BasePolicy
policy = BasePolicy.from_checkpoint(CKPT)
policy.to("cuda:0").eval()
To = policy.n_obs_steps
ports = policy.get_observation_ports()
dtype = policy.dtype
print(f"[diag] To={To} ports={ports} dtype={dtype} n_action_steps={policy.n_action_steps}")

z = zarr.open(ZARR, "r")
ends = z["meta/episode_ends"][:]
T = z["data/action"].shape[0]
act_ds = z["data/action"][:]
print(f"[diag] DATASET action  min={act_ds.min():.3f} max={act_ds.max():.3f} "
      f"mean={act_ds.mean():.3f} std={act_ds.std():.3f} shape={act_ds.shape}")
ap_ds = z["data/agent_pos"][:]
print(f"[diag] DATASET agent_pos min={ap_ds.min():.3f} max={ap_ds.max():.3f} "
      f"mean={ap_ds.mean():.3f} shape={ap_ds.shape}")
im = z["data/agentview_rgb"]
print(f"[diag] DATASET agentview_rgb dtype={im.dtype} min={int(np.asarray(im[0]).min())} "
      f"max={int(np.asarray(im[0]).max())} shape={im.shape}")

# pick ~8 diverse indices spread across episodes (avoid the very first frame of each ep)
starts = np.concatenate([[0], ends[:-1]])
idxs = []
for s, e in zip(starts, ends):
    if e - s > To + 5:
        idxs.append(int(s + (e - s) // 3))   # a mid-episode frame (has contact/motion)
    if len(idxs) >= 8:
        break
print(f"[diag] probing {len(idxs)} train windows at {idxs}")


def window(i):
    lo = max(0, i - To + 1)
    obs = {}
    for p in ports:
        arr = z[f"data/{p}"][lo:i + 1]                      # [<=To, ...]
        while arr.shape[0] < To:
            arr = np.concatenate([arr[:1], arr], axis=0)    # left-pad
        obs[p] = torch.from_numpy(np.asarray(arr)).to("cuda:0", dtype)[None]
    return obs


first_rows = []
with torch.inference_mode():
    for i in idxs:
        obs = window(i)
        res = policy.predict_action(obs, use_k_tokens=8, temperature=1.0, topk=1)  # greedy
        a = res["action"][0].detach().cpu().numpy()         # [R, 14]
        first_rows.append(a[0])
        print(f"[diag] idx {i:5d}  act[0]= " +
              " ".join(f"{v:+.2f}" for v in a[0]) +
              f"   (range {a.min():+.2f}..{a.max():+.2f})")

first_rows = np.stack(first_rows)                            # [n, 14]
per_dim_std = first_rows.std(axis=0)
print("\n[diag] per-dim STD of act[0] ACROSS the different train obs:")
print("       " + " ".join(f"{v:.3f}" for v in per_dim_std))
print(f"[diag] mean cross-obs std = {per_dim_std.mean():.4f}")
print("\n[VERDICT] cross-obs std >> 0 (e.g. > ~0.05) => policy IS obs-dependent -> sim failure is an"
      "\n          ADAPTER mismatch (B). ~0 => mean-collapse -> UNDERFIT (A).")
