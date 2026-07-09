#!/usr/bin/env python3
"""Patch RoboMimic raw HDF5 env_args for image extraction on an older robosuite stack.

Some MH raw datasets (e.g. demo_v15.hdf5 for can/square) were recorded with robosuite
1.5 metadata. Our cluster stack (robosuite 1.4 + robomimic 0.3) expects a different
env_args *schema* — notably a flat OSC_POSE controller block, not BASIC/body_parts.

This is a schema-compatibility migration for the reader environment, not a claim that
every can/square file needs the same edits. Pre-built image HDF5 (e.g. lift_mh_image.hdf5
from CDN) may already match the target stack and need no patch.

Trajectory data (states/actions) are untouched; only data.attrs['env_args'] is rewritten
so dataset_states_to_obs.py can recreate the sim for offscreen rendering.

Usage:
  python scripts/patch_robomimic_env_args.py \\
    --dataset data/robomimic/hdf5_datasets/can/mh/demo_v15.hdf5 \\
    --reference data/robomimic/hdf5_datasets/lift_mh_image.hdf5
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import h5py

# Keys present in robosuite 1.5 nested controllers but not used by our 1.4 OSC_POSE path.
_STRIP_FROM_OSC = frozenset({"gripper", "input_ref_frame"})

# env_kwargs fields copied from a working image HDF5 (lift) for 84x84 dual-camera extract.
_IMAGE_KWARGS_FROM_REF = (
    "has_renderer",
    "has_offscreen_renderer",
    "use_camera_obs",
    "camera_depths",
    "camera_heights",
    "camera_widths",
    "camera_names",
    "render_gpu_device_id",
)


def _load_env_args(hdf5_path: Path) -> dict:
    with h5py.File(hdf5_path, "r") as f:
        return json.loads(f["data"].attrs["env_args"])


def _flatten_controller(controller_configs: dict) -> dict:
    """BASIC/body_parts (1.5-style) -> flat OSC_POSE (1.4-style)."""
    if controller_configs.get("type") == "BASIC" and "body_parts" in controller_configs:
        parts = controller_configs["body_parts"]
        if "right" not in parts:
            raise ValueError("Expected body_parts.right in BASIC controller_configs")
        osc = copy.deepcopy(parts["right"])
        for key in _STRIP_FROM_OSC:
            osc.pop(key, None)
        if osc.get("type") != "OSC_POSE":
            raise ValueError(f"Unexpected nested controller type: {osc.get('type')}")
        return osc
    if controller_configs.get("type") == "OSC_POSE":
        osc = copy.deepcopy(controller_configs)
        for key in _STRIP_FROM_OSC:
            osc.pop(key, None)
        return osc
    raise ValueError(f"Unsupported controller_configs type: {controller_configs.get('type')}")


def patch_env_args(env_args: dict, reference_env_args: dict | None) -> dict:
    out = copy.deepcopy(env_args)
    out.pop("env_version", None)

    kw = out.setdefault("env_kwargs", {})
    kw.pop("lite_physics", None)

    kw["controller_configs"] = _flatten_controller(kw["controller_configs"])

    if reference_env_args is not None:
        ref_kw = reference_env_args["env_kwargs"]
        for key in _IMAGE_KWARGS_FROM_REF:
            if key in ref_kw:
                kw[key] = copy.deepcopy(ref_kw[key])

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Raw HDF5 to patch in place")
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Image HDF5 whose camera/render env_kwargs are copied (default: lift_mh_image next to dataset)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print patched JSON only")
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    reference = args.reference
    if reference is None:
        candidate = dataset.parent.parent.parent / "lift_mh_image.hdf5"
        reference = candidate if candidate.is_file() else None

    env_args = _load_env_args(dataset)
    ref_args = _load_env_args(reference) if reference is not None else None
    patched = patch_env_args(env_args, ref_args)

    if args.dry_run:
        print(json.dumps(patched, indent=2))
        return

    with h5py.File(dataset, "r+") as f:
        f["data"].attrs["env_args"] = json.dumps(patched)
    print(f"patched {dataset}")
    if reference is not None:
        print(f"  reference image kwargs from {reference}")


if __name__ == "__main__":
    main()
