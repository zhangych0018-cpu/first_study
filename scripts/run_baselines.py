"""Run DSG baseline search strategies."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sws_bo.optimization.baselines import lhs_search, random_search, surrogate_then_optimize, weighted_sum_bo
from sws_bo.problems import resolve_problem
from sws_bo.utils.mock_dsg_cst import MockDSGCSTSimulator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline comparisons")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    problem = resolve_problem("dsg")
    simulator = MockDSGCSTSimulator(seed=args.seed)
    output_dir = PROJECT_ROOT / "data" / "results" / "dsg_baselines"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "random": random_search(simulator, n_samples=30, seed=args.seed, problem=problem),
        "lhs": lhs_search(simulator, n_samples=30, seed=args.seed, problem=problem),
        "surrogate_then_optimize": surrogate_then_optimize(simulator, n_initial=20, seed=args.seed, problem=problem),
        "weighted_sum_bo": weighted_sum_bo(simulator, n_initial=20, n_iterations=3, seed=args.seed, problem=problem),
    }
    rows = []
    for name, payload in results.items():
        if "raw" in payload:
            feasible = payload["raw"][:, 3] <= problem.s11_constraint_db
            rows.append({"method": name, "n_points": len(payload["raw"]), "feasible_rate": float(feasible.mean())})
        else:
            rows.append({"method": name, "n_points": 1, "feasible_rate": float(payload["best_result"]["S11_max"] <= problem.s11_constraint_db)})
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "baseline_summary.csv", index=False, encoding="utf-8")
    print(output_dir / "baseline_summary.csv")


if __name__ == "__main__":
    main()
