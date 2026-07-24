"""
Offline (NO sim) check: does the policy CONDITION its APPROACH-phase action on the observation
(i.e. track the cube), or does it collapse to a near-constant action ("moves left regardless")?

Idea: in the demos, the aim/approach phase (2nd half of each episode) varies with the cube
position -> GT action varies across episodes. If the policy TRACKS that variation, its prediction
correlates with GT (conditions on obs -> rollout failure is compounding/early -> more epochs help).
If the policy predicts a near-CONSTANT action while GT varies, it's obs-collapsed on the critical
phase (under-fit -> the thing the user saw in the video).

Run on docker (conda python). Auto-finds the newest policy ckpt for TASK unless CKPT is set:
  cd oat && source robotwin_pipeline/config.sh
  TASK=beat_block_hammer TASK_CFG=beat_block_hammer NDEMO=330 ${OATPY} robotwin_pipeline/diag_track.py
"""
import os, glob, numpy as np, torch, zarr

ZARR = os.environ["ZARR"]
CKPT = os.environ.get("CKPT", "")
if not CKPT:
    runs = sorted(glob.glob(f"{os.environ['OAT_DIR']}/output/*/*oatpolicy_{os.environ['TASK']}*"),
                  key=os.path.getmtime)
    assert runs, "no policy run found; set CKPT=..."
    cand = f"{runs[-1]}/checkpoints/latest.ckpt"
    CKPT = cand if os.path.exists(cand) else sorted(
        glob.glob(f"{runs[-1]}/checkpoints/ep-*.ckpt"), key=os.path.getmtime)[-1]
print(f"[diag] ckpt = {CKPT}")

from oat.policy.base_policy import BasePolicy
policy = BasePolicy.from_checkpoint(CKPT); policy.to("cuda:0").eval()
To = policy.n_obs_steps; ports = policy.get_observation_ports(); dtype = policy.dtype

z = zarr.open(ZARR, "r")
ends = z["meta/episode_ends"][:]
starts = np.concatenate([[0], ends[:-1]])
act_ds = z["data/action"][:]
print(f"[diag] To={To} ports={ports} n_action_steps={policy.n_action_steps} eps={len(ends)}")
print(f"[diag] dataset action std per-dim = " + " ".join(f"{v:.2f}" for v in act_ds.std(0)))


def window(i):
    lo = max(0, i - To + 1)
    obs = {}
    for p in ports:
        arr = np.asarray(z[f"data/{p}"][lo:i + 1])
        while arr.shape[0] < To:
            arr = np.concatenate([arr[:1], arr], axis=0)
        obs[p] = torch.from_numpy(arr).to("cuda:0", dtype)[None]
    return obs


# one APPROACH-phase state per episode (70% through -> after grasp, aiming at the cube)
idxs = []
for s, e in zip(starts, ends):
    L = e - s
    if L > To + 5:
        idxs.append(int(s + 0.70 * L))
print(f"[diag] {len(idxs)} approach-phase states (70% into each episode)")

preds, gts = [], []
with torch.inference_mode():
    for i in idxs:
        res = policy.predict_action(window(i), use_k_tokens=8, temperature=1.0, topk=1)  # greedy
        preds.append(res["action"][0, 0].detach().cpu().numpy())   # [14]
        gts.append(np.asarray(z["data/action"][i]))                # [14]
preds = np.stack(preds); gts = np.stack(gts)                       # [N,14]

err = np.abs(preds - gts).mean()
gt_std = gts.std(0)                                                # how much the demo action VARIES across cubes
pred_std = preds.std(0)                                            # does the policy VARY too?
# per-dim correlation pred<->GT across the approach states
corr = np.array([
    np.corrcoef(preds[:, d], gts[:, d])[0, 1] if gt_std[d] > 1e-4 else np.nan
    for d in range(14)
])
ratio = pred_std / (gt_std + 1e-6)

print(f"\n[diag] mean |pred-GT| (approach) = {err:.3f}   (dataset action std {act_ds.std():.3f})")
print("[diag] per-dim   GT_std : " + " ".join(f"{v:.2f}" for v in gt_std))
print("[diag] per-dim pred_std : " + " ".join(f"{v:.2f}" for v in pred_std))
print("[diag] per-dim  ratio   : " + " ".join(f"{v:.2f}" for v in ratio))
print("[diag] per-dim  corr    : " + " ".join(f"{'nan' if np.isnan(v) else f'{v:+.2f}'}" for v in corr))
# focus on the dims that actually vary in the demos (the approach/aim dims)
varying = gt_std > 0.15
mc = np.nanmean(corr[varying]); mr = np.nanmean(ratio[varying])
print(f"\n[diag] on VARYING dims (GT_std>0.15, n={int(varying.sum())}): mean corr={mc:+.2f}  mean std-ratio={mr:.2f}")
print("\n[VERDICT]")
print("  corr HIGH (>~0.5) & ratio ~1  => policy TRACKS the cube-varying demo -> it CONDITIONS on obs")
print("     -> rollout 'moves left regardless' is COMPOUNDING/early -> MORE EPOCHS should fix it.")
print("  corr ~0 & ratio <<1 (pred near-constant while GT varies) => OBS-COLLAPSE on the approach")
print("     -> critical aim-phase is under-fit (drowned by easy grasp/carry tokens) -> needs more")
print("        epochs / up-weighting the aim steps / more demos; won't fix by waiting alone.")
