#!/usr/bin/env python3
"""Validate HF robomimic_zarr + robomimic-oattok-policy against cluster originals."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import zarr
from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

DS = "hackhackhack66666/robomimic_zarr"
MODEL = "hackhackhack66666/robomimic-oattok-policy"
TMP = Path("/tmp/hf_rm_validate")

ORIG = {
    "tokenizers/lift/ep-1970_mse-0.006.ckpt": "output/20260704/203215_train_oattok_lift_N200/checkpoints/ep-1970_mse-0.006.ckpt",
    "tokenizers/can/ep-0520_mse-0.005.ckpt": "output/20260705/210939_train_oattok_can_N200/checkpoints/ep-0520_mse-0.005.ckpt",
    "tokenizers/square/ep-0690_mse-0.004.ckpt": "output/20260706/005048_train_oattok_square_N200/checkpoints/ep-0690_mse-0.004.ckpt",
    "policies/lift/ep-0900_sr-0.930.ckpt": "output/20260719/144024_train_oatpolicy_lift_N200/checkpoints/ep-0900_sr-0.930.ckpt",
    "policies/can/ep-1700_sr-0.940.ckpt": "output/20260706/173343_train_oatpolicy_can_N200/checkpoints/ep-1700_sr-0.940.ckpt",
    "policies/square/ep-0700_sr-0.420.ckpt": "output/20260720/215024_train_oatpolicy_square_N200/checkpoints/ep-0700_sr-0.420.ckpt",
    "awr/awr_s10000_lift.ckpt": "my_models/awr_s10000_lift.ckpt",
    "awr/awr_s10000_can.ckpt": "my_models/awr_s10000_can.ckpt",
    "awr/awr_s10000_square.ckpt": "my_models/awr_s10000_square.ckpt",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def zarr_stats(path: Path) -> dict:
    r = zarr.open(str(path), "r")
    ends = np.asarray(r["meta"]["episode_ends"][:])
    a = r["data"]["action"]
    rgb = r["data"]["agentview_rgb"]
    img = np.asarray(rgb[0])
    a0 = np.asarray(a[0:64])
    assert len(ends) == 200, (path, len(ends))
    assert int(ends[-1]) == int(a.shape[0])
    assert np.all(np.diff(ends) > 0)
    assert np.isfinite(a0).all()
    assert img.dtype == np.uint8 and img.shape[-1] == 3
    return {
        "n_ep": int(len(ends)),
        "n_steps": int(a.shape[0]),
        "action_shape": list(a.shape),
        "rgb_shape": list(rgb.shape),
    }


def count_tensors(obj) -> tuple[int, int]:
    n_t = bad = 0
    stack = [obj]
    while stack:
        d = stack.pop()
        if not isinstance(d, dict):
            continue
        for v in d.values():
            if torch.is_tensor(v):
                n_t += 1
                if v.is_floating_point() and v.numel() and not torch.isfinite(v).all():
                    bad += 1
            elif isinstance(v, dict):
                stack.append(v)
    return n_t, bad


def main() -> None:
    TMP.mkdir(exist_ok=True)
    print("=== ZARR from HF ===")
    zroot = Path(
        snapshot_download(
            DS,
            repo_type="dataset",
            local_dir=str(TMP / "zarr"),
            allow_patterns=[
                "lift_N200.zarr/**",
                "can_N200.zarr/**",
                "square_N200.zarr/**",
            ],
        )
    )
    for name in ["lift_N200.zarr", "can_N200.zarr", "square_N200.zarr"]:
        hf_p = zroot / name
        ref_p = Path("data/robomimic") / name
        hf_s = zarr_stats(hf_p)
        ref_s = zarr_stats(ref_p)
        assert hf_s == ref_s, (name, hf_s, ref_s)
        r_hf = zarr.open(str(hf_p), "r")
        r_ref = zarr.open(str(ref_p), "r")
        assert np.allclose(
            np.asarray(r_hf["data"]["action"][0]),
            np.asarray(r_ref["data"]["action"][0]),
        )
        assert np.array_equal(
            np.asarray(r_hf["data"]["agentview_rgb"][0]),
            np.asarray(r_ref["data"]["agentview_rgb"][0]),
        )
        assert np.array_equal(
            np.asarray(r_hf["meta"]["episode_ends"][:]),
            np.asarray(r_ref["meta"]["episode_ends"][:]),
        )
        print("MATCH", name, hf_s)

    print("=== CKPTS from HF ===")
    mfiles = set(list_repo_files(MODEL, repo_type="model"))
    for p in ORIG:
        assert p in mfiles, f"missing {p}"
    assert not any("1400" in f for f in mfiles)

    ckpt_dir = TMP / "ckpt"
    for p, src in ORIG.items():
        local = Path(
            hf_hub_download(MODEL, p, repo_type="model", local_dir=str(ckpt_dir))
        )
        obj = torch.load(local, map_location="cpu", weights_only=False)
        assert isinstance(obj, dict)
        n_t, bad = count_tensors(obj)
        assert n_t > 100 and bad == 0, (p, n_t, bad)
        h_hf, h_src = sha256(local), sha256(Path(src))
        assert h_hf == h_src, (p, h_hf, h_src)
        print(f"SHA_OK {p}: tensors={n_t}")

    print("\nALL VALID: zarr content matches cluster; ckpt sha256 match")


if __name__ == "__main__":
    main()
