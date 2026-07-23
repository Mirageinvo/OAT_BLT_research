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


RGB_PORTS = [p for p in ports if "rgb" in p]


def window(i, swap_rgb=False):
    lo = max(0, i - To + 1)
    obs = {}
    for p in ports:
        arr = np.asarray(z[f"data/{p}"][lo:i + 1])          # [<=To, ...]
        while arr.shape[0] < To:
            arr = np.concatenate([arr[:1], arr], axis=0)    # left-pad
        if swap_rgb and p in RGB_PORTS:
            arr = arr[..., ::-1].copy()                     # reverse channels (RGB<->BGR)
        obs[p] = torch.from_numpy(arr).to("cuda:0", dtype)[None]
    return obs


def run(swap_rgb):
    rows = []
    with torch.inference_mode():
        for i in idxs:
            res = policy.predict_action(window(i, swap_rgb), use_k_tokens=8,
                                        temperature=1.0, topk=1)  # greedy
            rows.append(res["action"][0].detach().cpu().numpy()[0])
    rows = np.stack(rows)
    return rows, rows.std(axis=0).mean()


print("\n=== NORMAL channel order (as stored in Zarr = training convention) ===")
rows_n, std_n = run(False)
for i, r in zip(idxs, rows_n):
    print(f"[diag] idx {i:5d}  act[0]= " + " ".join(f"{v:+.2f}" for v in r))
print(f"[diag] mean cross-obs std (NORMAL)  = {std_n:.4f}")

print("\n=== SWAPPED channel order (RGB<->BGR) — simulates a live/dataset channel mismatch ===")
rows_s, std_s = run(True)
for i, r in zip(idxs, rows_s):
    print(f"[diag] idx {i:5d}  act[0]= " + " ".join(f"{v:+.2f}" for v in r))
print(f"[diag] mean cross-obs std (SWAPPED) = {std_s:.4f}")

print(f"\n[VERDICT] NORMAL std={std_n:.4f}  SWAPPED std={std_s:.4f}")
print("  - SWAPPED std collapses (<< NORMAL, ~0) => policy is CHANNEL-SENSITIVE. If the live sim feeds")
print("    the OTHER channel order than training, THAT is the sim-failure bug -> swap channels in the adapter.")
print("  - SWAPPED std ~= NORMAL => channels are NOT the issue => sim failure is UNDERFIT / distribution-shift (A).")
