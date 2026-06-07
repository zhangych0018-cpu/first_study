"""本脚本用于为 DSG 问题批量生成初始样本数据，通常服务于 mock 演示、代理模型预热和手工检查输入输出分布。它会按照当前问题定义和采样策略生成可直接保存的数据表。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sws_bo.geometry.dsg_sws import validate_dsg_geometry
from sws_bo.optimization.initial_design import generate_hybrid_design
from sws_bo.problems import resolve_problem
from sws_bo.utils.mock_dsg_cst import MockDSGCSTSimulator


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate initial SWS design data")
    parser.add_argument("--backend", choices=["mock_dsg", "dry_run"], default=None)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    problem = resolve_problem("dsg")
    backend = args.backend or "mock_dsg"
    output = args.output or "data/processed/dsg_initial_mock_data.csv"
    validator = lambda point, _: validate_dsg_geometry(point).is_valid
    design = generate_hybrid_design(args.n_samples, seed=args.seed, normalized=False, problem=problem, validator=validator)

    rows = []
    simulator = MockDSGCSTSimulator(seed=args.seed) if backend == "mock_dsg" else None

    for point in design.valid:
        row = {name: float(point[idx]) for idx, name in enumerate(problem.param_names)}
        if simulator is not None:
            result = simulator.run(point)
            row.update(result.to_dict())
        else:
            row.update({"success": True, "failure_reason": None, "sim_time": 0.0})
        rows.append(row)
    output_path = PROJECT_ROOT / output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
