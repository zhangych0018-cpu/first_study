"""本模块实现 Pareto 有效性判定、可行前沿提取和代表性设计选择逻辑，用于把多目标结果整理成工程上更容易阅读的候选集合。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def is_pareto_efficient(objectives: np.ndarray) -> np.ndarray:
    """返回最小化问题下的 Pareto 有效掩码，用于识别哪些点没有被其他点同时全面优于。"""

    values = np.asarray(objectives, dtype=float)
    n = len(values)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        dominating = np.all(values <= values[i], axis=1) & np.any(values < values[i], axis=1)
        if np.any(dominating):
            mask[i] = False
    return mask


def feasible_pareto_front(objectives: np.ndarray, feasible_mask: np.ndarray) -> np.ndarray:
    """在仅保留可行样本的前提下提取 Pareto 前沿，避免不可行设计混入代表结果。"""

    mask = np.zeros(len(objectives), dtype=bool)
    feasible_idx = np.where(np.asarray(feasible_mask, dtype=bool))[0]
    if len(feasible_idx) == 0:
        return mask
    feasible_obj = np.asarray(objectives, dtype=float)[feasible_idx]
    pareto = is_pareto_efficient(feasible_obj)
    mask[feasible_idx[pareto]] = True
    return mask


def select_representative_designs(
    pareto_df: pd.DataFrame,
    n_select: int = 4,
    kc_col: str = "Kc_mean",
    vp_col: str = "vp_std",
    loss_col: str = "ohmic_loss_mean",
) -> pd.DataFrame:
    """从 Pareto 数据表中挑选具有代表性的多样化候选，便于工程师后续人工比较和筛选。"""

    if len(pareto_df) <= n_select:
        return pareto_df.copy()
    df = pareto_df.copy()
    indices = [df[kc_col].idxmax(), df[vp_col].idxmin(), df[loss_col].idxmin()]
    remaining = [idx for idx in df.index if idx not in indices]
    if remaining:
        knee_idx = df.loc[remaining, [kc_col, vp_col, loss_col]].rank(pct=True).sum(axis=1).idxmax()
        indices.append(knee_idx)
    return df.loc[list(dict.fromkeys(indices))].reset_index(drop=True)
