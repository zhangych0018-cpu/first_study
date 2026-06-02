"""Sensitivity utilities based on ARD lengthscales."""

from __future__ import annotations

import pandas as pd

from ..problems.dsg_bwo_problem import DSGSWSProblem
from ..problems import resolve_problem
from ..surrogate.independent_gp import get_ard_lengthscales


def ard_lengthscale_to_sensitivity(lengthscales: dict[str, dict[str, float]], problem=DSGSWSProblem) -> pd.DataFrame:
    """Convert ARD lengthscales to normalized inverse-scale sensitivities."""

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
    """Rank parameters by sensitivity per objective."""

    df = sensitivity_df.copy()
    df["rank"] = df.groupby("output")["sensitivity"].rank(ascending=False, method="dense")
    return df.sort_values(["output", "rank", "parameter"]).reset_index(drop=True)


def export_sensitivity_table(model, path, problem=DSGSWSProblem) -> pd.DataFrame:
    """Export ARD sensitivity table from a fitted GP model."""

    table = rank_parameters_by_objective(ard_lengthscale_to_sensitivity(get_ard_lengthscales(model), problem=problem))
    table.to_csv(path, index=False, encoding="utf-8")
    return table
