"""Engineering proxy rules for validating DSG candidate geometries."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from ..geometry.dsg_sws import validate_dsg_geometry
from ..problems.dsg_bwo_problem import DSGSWSProblem


def _to_array(x: dict | Iterable[float] | np.ndarray) -> np.ndarray:
    if isinstance(x, dict):
        return np.array([x[name] for name in DSGSWSProblem.param_names], dtype=float)
    return np.asarray(x, dtype=float)


def check_bounds(x: dict | Iterable[float] | np.ndarray) -> bool:
    """Check physical bounds only."""

    arr = _to_array(x)
    return DSGSWSProblem.validate_x(arr)


def check_basic_geometry(x: dict | Iterable[float] | np.ndarray) -> bool:
    """Check coarse DSG geometry relations."""

    w, p, t, g, h = _to_array(x)
    return bool(w > 0 and p > 0 and t > 0 and g > 0 and h > 0 and g < p and t < h + 0.25)


def check_channel_clearance(x: dict | Iterable[float] | np.ndarray) -> bool:
    """Ensure the beam tunnel is not unrealistically narrow."""

    _, _, t, g, h = _to_array(x)
    return bool(t >= 0.16 and h - t >= 0.05 and g <= 0.75 * h)


def check_self_intersection_proxy(x: dict | Iterable[float] | np.ndarray) -> bool:
    """Proxy rule for likely over-packed DSG geometry."""

    w, p, t, g, h = _to_array(x)
    packedness = (g / max(p, 1e-9)) + 0.4 * (h / max(w, 1e-9)) + 0.35 * (t / max(h, 1e-9))
    return bool(packedness <= 1.05)


def is_valid_design(x: dict | Iterable[float] | np.ndarray) -> bool:
    """Combined validity predicate used by design generators and simulators."""

    return (
        check_bounds(x)
        and check_basic_geometry(x)
        and check_channel_clearance(x)
        and check_self_intersection_proxy(x)
        and validate_dsg_geometry(_to_array(x)).is_valid
    )


def repair_or_reject_design(
    x: dict | Iterable[float] | np.ndarray,
    max_tries: int = 3,
) -> np.ndarray | None:
    """Try simple clipping and reference blending before rejecting."""

    arr = np.clip(_to_array(x), DSGSWSProblem.bounds[:, 0], DSGSWSProblem.bounds[:, 1])
    if is_valid_design(arr):
        return arr

    candidate = arr.copy()
    for blend in np.linspace(0.15, 0.55, max_tries):
        candidate = (1.0 - blend) * candidate + blend * DSGSWSProblem.reference_design
        candidate = np.clip(candidate, DSGSWSProblem.bounds[:, 0], DSGSWSProblem.bounds[:, 1])
        if is_valid_design(candidate):
            return candidate
    return None
