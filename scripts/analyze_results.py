"""本脚本负责读取一次或多次 DSG 优化运行产生的结果目录，并自动生成 Pareto 图、超体积历史、约束可行性和敏感性等分析输出。它是结果复盘与图表整理的统一入口。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sws_bo.analysis.hypervolume import hypervolume_history
from sws_bo.analysis.plots import (
    plot_ard_sensitivity,
    plot_constraint_feasibility,
    plot_hypervolume_history,
    plot_pareto_2d,
    plot_pareto_3d,
)
from sws_bo.analysis.sensitivity import export_sensitivity_table
from sws_bo.problems import resolve_problem
from sws_bo.surrogate.train import train_independent_gp


def _kc_column(df: pd.DataFrame) -> str:
    return "Kc_TM21_mean" if "Kc_TM21_mean" in df.columns else "Kc_mean"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze BO results")
    parser.add_argument("--results-dir", type=str, default="data/results/dsg_mock_demo")
    args = parser.parse_args()

    results_dir = PROJECT_ROOT / args.results_dir
    eval_df = pd.read_csv(results_dir / "evaluations.csv")
    history_df = pd.read_csv(results_dir / "history.csv")
    figures_dir = results_dir / "figures"
    problem = resolve_problem("dsg")
    successful = eval_df[eval_df["success"]].reset_index(drop=True)
    hv_df = hypervolume_history(successful, problem.hypervolume_ref_point, ["neg_Kc_mean", "vp_std", "ohmic_loss_mean"])
    kc_col = _kc_column(successful)
    plot_pareto_2d(successful, figures_dir / "analysis_pareto_2d.png", x_col="ohmic_loss_mean", y_col=kc_col)
    feasible = successful[successful["is_feasible"]]
    if len(feasible):
        plot_pareto_3d(feasible, figures_dir / "analysis_pareto_3d.png", x_col="vp_std", y_col="ohmic_loss_mean", z_col=kc_col)
    plot_hypervolume_history(hv_df, figures_dir / "analysis_hv.png")
    plot_constraint_feasibility(history_df, figures_dir / "analysis_feasibility.png")

    if len(successful) >= 4:
        train_X = torch.tensor(problem.normalize(successful[problem.param_names].to_numpy()), dtype=torch.double)
        train_Y = torch.tensor(successful[problem.target_names].to_numpy(), dtype=torch.double)
        model = train_independent_gp(train_X, train_Y, ard=True)
        sensitivity_table = export_sensitivity_table(model, results_dir / "analysis_sensitivity.csv", problem=problem)
        plot_ard_sensitivity(sensitivity_table, figures_dir / "analysis_ard_sensitivity.png")
    print(figures_dir)


if __name__ == "__main__":
    main()
