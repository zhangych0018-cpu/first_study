"""本脚本用于快速跑通 DSG 的 mock 演示流程，覆盖初始化采样、代理建模、采集优化、结果分析等关键链路。它强调速度和可复现性，适合本地自检和演示。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

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
from sws_bo.optimization.bo_loop import SWSBayesianOptimizer
from sws_bo.problems import resolve_problem
from sws_bo.surrogate.train import train_independent_gp


def _kc_column(df) -> str:
    return "Kc_TM21_mean" if "Kc_TM21_mean" in df.columns else "Kc_mean"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a mock BO demo")
    parser.add_argument("--n-initial", type=int, default=20)
    parser.add_argument("--n-iterations", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pf-min", type=float, default=0.85)
    args = parser.parse_args()

    problem = resolve_problem("dsg")
    output_dir = PROJECT_ROOT / "data" / "results" / "dsg_mock_demo"
    optimizer = SWSBayesianOptimizer(
        backend="mock_dsg",
        problem=problem,
        n_initial=args.n_initial,
        n_iterations=args.n_iterations,
        batch_size=args.batch_size,
        seed=args.seed,
        pf_min=args.pf_min,
        output_dir=output_dir,
    )
    optimizer.run()
    successful = optimizer._successful_evaluations()
    pareto = optimizer.export_pareto_dataframe()
    history = hypervolume_history(
        successful,
        ref_point=problem.hypervolume_ref_point,
        objective_columns=["neg_Kc_mean", "vp_std", "ohmic_loss_mean"],
    )
    figures_dir = output_dir / "figures"
    kc_col = _kc_column(successful)
    plot_pareto_2d(successful, figures_dir / "pareto_2d.png", x_col="ohmic_loss_mean", y_col=kc_col)
    if len(pareto):
        plot_pareto_3d(pareto, figures_dir / "pareto_3d.png", x_col="vp_std", y_col="ohmic_loss_mean", z_col=kc_col)
    plot_hypervolume_history(history, figures_dir / "hypervolume_history.png")
    plot_constraint_feasibility(optimizer.history, figures_dir / "feasibility_history.png")

    if len(successful) >= 4:
        train_X = torch.tensor(problem.normalize(successful[problem.param_names].to_numpy()), dtype=torch.double)
        train_Y = torch.tensor(successful[problem.target_names].to_numpy(), dtype=torch.double)
        model = train_independent_gp(train_X, train_Y, ard=True)
        sensitivity_table = export_sensitivity_table(model, output_dir / "sensitivity.csv", problem=problem)
        plot_ard_sensitivity(sensitivity_table, figures_dir / "ard_sensitivity.png")

    print(output_dir)


if __name__ == "__main__":
    main()
