"""本模块提供 DSG 及相关几何设计的工程代理检查规则，用于在进入仿真前尽早发现越界、过窄通道和潜在自交等明显问题。"""

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
    """仅检查设计参数是否落在物理边界之内，是几何检查链路中最基础的一层。"""

    arr = _to_array(x)
    return DSGSWSProblem.validate_x(arr)


def check_basic_geometry(x: dict | Iterable[float] | np.ndarray) -> bool:
    """检查 DSG 几何的粗粒度关系约束，例如槽厚与周期、结构高度等基本组合是否合法。"""

    w, p, t, g, h = _to_array(x)
    return bool(w > 0 and p > 0 and t > 0 and g > 0 and h > 0 and g < p and t < h + 0.25)


def check_channel_clearance(x: dict | Iterable[float] | np.ndarray) -> bool:
    """检查电子束通道是否过窄，从工程经验上避免明显不可制造或不可工作的设计。"""

    _, _, t, g, h = _to_array(x)
    return bool(t >= 0.16 and h - t >= 0.05 and g <= 0.75 * h)


def check_self_intersection_proxy(x: dict | Iterable[float] | np.ndarray) -> bool:
    """用代理规则近似判断 DSG 结构是否过于拥挤，从而提前发现潜在自交或过密堆叠问题。"""

    w, p, t, g, h = _to_array(x)
    packedness = (g / max(p, 1e-9)) + 0.4 * (h / max(w, 1e-9)) + 0.35 * (t / max(h, 1e-9))
    return bool(packedness <= 1.05)


def is_valid_design(x: dict | Iterable[float] | np.ndarray) -> bool:
    """把多项几何检查组合成一个统一有效性判定函数，供采样器和仿真器直接调用。"""

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
    """在正式拒绝样本前尝试简单修复，例如裁剪或向参考设计回拉，以提高初始采样可用率。"""

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
