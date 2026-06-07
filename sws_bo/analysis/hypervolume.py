"""本模块实现可行解超体积及其历史曲线计算，是评估多目标优化效率和收敛趋势的重要工具。它强调结果可解释性和指标复现性。"""

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
    """对低维目标数据计算超体积指标，用于量化当前可行 Pareto 前沿相对于参考点的覆盖质量。"""

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
    """按评估历史逐步累计计算超体积，帮助观察多目标优化在运行过程中的收敛趋势。"""

    rows = []
    for step in range(1, len(history_df) + 1):
        subset = history_df.iloc[:step]
        feasible = subset[subset[feasible_column]]
        hv = compute_hypervolume(feasible[objective_columns].to_numpy(), ref_point, maximize=False)
        rows.append({"step": step, "hypervolume": hv})
    return pd.DataFrame(rows)
