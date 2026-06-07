"""本模块依据 ARD 长度尺度等信息估计参数敏感性，并生成排序表和可视化数据。它帮助研究者理解 DSG 结构中哪些几何变量更关键。"""

from __future__ import annotations

import pandas as pd

from ..problems.dsg_bwo_problem import DSGSWSProblem
from ..problems import resolve_problem
from ..surrogate.independent_gp import get_ard_lengthscales


def ard_lengthscale_to_sensitivity(lengthscales: dict[str, dict[str, float]], problem=DSGSWSProblem) -> pd.DataFrame:
    """把 ARD 长度尺度转换成归一化的反尺度敏感性指标，使数值更便于跨参数比较。"""

    problem = resolve_problem(problem)
    rows = []
    for output_name, mapping in lengthscales.items():
        inv = {name: 1.0 / max(value, 1e-9) for name, value in mapping.items()}
        total = sum(inv.values()) or 1.0
        for idx, param_name in enumerate(problem.param_names):
            key = f"x{idx}"
            rows.append(
                {
                    "output": output_name,
                    "parameter": param_name,
                    "lengthscale": mapping.get(key, float("nan")),
                    "sensitivity": inv.get(key, 0.0) / total,
                }
            )
    return pd.DataFrame(rows)


def rank_parameters_by_objective(sensitivity_df: pd.DataFrame) -> pd.DataFrame:
    """针对每个目标输出按敏感性大小对参数排序，帮助识别哪些几何变量更主导结果变化。"""

    df = sensitivity_df.copy()
    df["rank"] = df.groupby("output")["sensitivity"].rank(ascending=False, method="dense")
    return df.sort_values(["output", "rank", "parameter"]).reset_index(drop=True)


def export_sensitivity_table(model, path, problem=DSGSWSProblem) -> pd.DataFrame:
    """从已训练好的 GP 中提取 ARD 信息并导出成敏感性表格，便于报告和绘图使用。"""

    table = rank_parameters_by_objective(ard_lengthscale_to_sensitivity(get_ard_lengthscales(model), problem=problem))
    table.to_csv(path, index=False, encoding="utf-8")
    return table
