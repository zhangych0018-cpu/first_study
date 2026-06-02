"""Run DSG ablation experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sws_bo.analysis.ablation import compare_acquisition, compare_ard, compare_constraint_handling, compare_initial_sample_sizes, compare_robust
from sws_bo.problems import resolve_problem


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    problem = resolve_problem("dsg")
    output_dir = PROJECT_ROOT / "data" / "results" / "dsg_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_sizes = [50, 100] if args.quick else [50, 100, 150, 200]
    compare_initial_sample_sizes(sample_sizes, problem=problem).to_csv(output_dir / "initial_sample_sizes.csv", index=False, encoding="utf-8")
    compare_acquisition(problem=problem).to_csv(output_dir / "acquisition.csv", index=False, encoding="utf-8")
    compare_constraint_handling(problem=problem).to_csv(output_dir / "constraint_handling.csv", index=False, encoding="utf-8")
    compare_ard(problem=problem).to_csv(output_dir / "ard.csv", index=False, encoding="utf-8")
    compare_robust(problem=problem).to_csv(output_dir / "robust.csv", index=False, encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
