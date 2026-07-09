"""Robosuite 1.4 compatibility for env metadata stored in RoboMimic HDF5 (mh image)."""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

# robosuite 1.5+ env_kwargs not accepted by 1.4.
_RS14_STRIP_ENV_KWARGS = frozenset({"lite_physics"})

_STRIP_FROM_OSC = frozenset({"gripper", "input_ref_frame"})

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


def flatten_controller(controller_configs: dict) -> dict:
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
    return copy.deepcopy(controller_configs)


def sanitize_env_meta(
    env_meta: dict,
    reference_env_meta: Optional[dict] = None,
) -> dict:
    """Make dataset env metadata constructible on robosuite 1.4 + robomimic 0.3."""
    meta = copy.deepcopy(env_meta)
    meta.pop("env_version", None)

    kw: Dict[str, Any] = meta.setdefault("env_kwargs", {})
    for key in _RS14_STRIP_ENV_KWARGS:
        kw.pop(key, None)

    if "controller_configs" in kw:
        kw["controller_configs"] = flatten_controller(kw["controller_configs"])

    if reference_env_meta is not None:
        ref_kw = reference_env_meta.get("env_kwargs", {})
        for key in _IMAGE_KWARGS_FROM_REF:
            if key in ref_kw:
                kw[key] = copy.deepcopy(ref_kw[key])

    return meta
