"""Hypervolume utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from botorch.utils.multi_objective.hypervolume import Hypervolume


def compute_hypervolume(
    objective_values: np.ndarray,
    ref_point: np.ndarray,
    maximize: bool = False,
) -> float:
    """Compute hypervolume for low-dimensional objective data."""

    values = np.asarray(objective_values, dtype=float)
    ref = np.asarray(ref_point, dtype=float)
    if len(values) == 0:
        return 0.0
    if not maximize:
        values = -values
        ref = -ref
    hv = Hypervolume(torch.tensor(ref, dtype=torch.double))
    return float(hv.compute(torch.tensor(values, dtype=torch.double)))


def hypervolume_history(
    history_df: pd.DataFrame,
    ref_point: np.ndarray,
    objective_columns: list[str],
    feasible_column: str = "is_feasible",
) -> pd.DataFrame:
    """Compute cumulative hypervolume over evaluation history."""

    rows = []
    for step in range(1, len(history_df) + 1):
        subset = history_df.iloc[:step]
        feasible = subset[subset[feasible_column]]
        hv = compute_hypervolume(feasible[objective_columns].to_numpy(), ref_point, maximize=False)
        rows.append({"step": step, "hypervolume": hv})
    return pd.DataFrame(rows)
