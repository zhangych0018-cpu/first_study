"""Pareto-front utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def is_pareto_efficient(objectives: np.ndarray) -> np.ndarray:
    """Return the minimization Pareto mask."""

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
    """Pareto mask restricted to feasible rows."""

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
    """Select diverse representative designs from a Pareto dataframe."""

    if len(pareto_df) <= n_select:
        return pareto_df.copy()
    df = pareto_df.copy()
    indices = [df[kc_col].idxmax(), df[vp_col].idxmin(), df[loss_col].idxmin()]
    remaining = [idx for idx in df.index if idx not in indices]
    if remaining:
        knee_idx = df.loc[remaining, [kc_col, vp_col, loss_col]].rank(pct=True).sum(axis=1).idxmax()
        indices.append(knee_idx)
    return df.loc[list(dict.fromkeys(indices))].reset_index(drop=True)
