"""本模块封装 DSG 项目的消融实验流程，负责搭建对比设置、执行实验并汇总评价指标。它帮助系统性回答哪些设计选择真正有效。"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..optimization.baselines import weighted_sum_bo
from ..optimization.bo_loop import SWSBayesianOptimizer
from ..optimization.robust_objective import robust_candidate_ranking
from ..problems.dsg_bwo_problem import DSGSWSProblem
from ..problems import resolve_problem


def compare_initial_sample_sizes(sample_sizes: list[int], seed: int = 42, problem=DSGSWSProblem) -> pd.DataFrame:
    """比较不同初始样本预算下的 BO 表现，用于观察起始数据量对收敛速度和最终结果的影响。"""

    problem = resolve_problem(problem)
    rows = []
    for size in sample_sizes:
        optimizer = SWSBayesianOptimizer(backend="mock_dsg", problem=problem, n_initial=size, n_iterations=2, batch_size=1, seed=seed + size)
        result = optimizer.run()
        rows.append({"initial_samples": size, "hypervolume": result["best_hypervolume"], "feasible_rate": result["feasible_rate"]})
    return pd.DataFrame(rows)


def compare_acquisition(seed: int = 42, problem=DSGSWSProblem) -> pd.DataFrame:
    """对比 qNEHVI 和加权和基线等不同采集策略的效果，帮助判断采集函数选择是否真的带来收益。"""

    problem = resolve_problem(problem)
    bo = SWSBayesianOptimizer(backend="mock_dsg", problem=problem, n_initial=20, n_iterations=2, batch_size=1, seed=seed)
    bo_result = bo.run()
    baseline = weighted_sum_bo(bo.simulator, n_initial=20, n_iterations=2, seed=seed, problem=problem)
    return pd.DataFrame(
        [
            {"method": "qNEHVI", "hypervolume": bo_result["best_hypervolume"], "wall_clock_time": bo_result["wall_clock_time"]},
            {"method": "weighted_sum", "hypervolume": np.nan, "wall_clock_time": np.nan if "history" not in baseline else len(baseline["history"])},
        ]
    )


def compare_constraint_handling(seed: int = 42, problem=DSGSWSProblem) -> pd.DataFrame:
    """比较带可行概率过滤和不带过滤两种约束处理方式，分析约束策略对搜索稳定性的影响。"""

    problem = resolve_problem(problem)
    rows = []
    for pf_min in [0.0, 0.9]:
        optimizer = SWSBayesianOptimizer(backend="mock_dsg", problem=problem, n_initial=20, n_iterations=2, batch_size=1, seed=seed, pf_min=pf_min)
        result = optimizer.run()
        rows.append({"pf_min": pf_min, "hypervolume": result["best_hypervolume"], "feasible_rate": result["feasible_rate"]})
    return pd.DataFrame(rows)


def compare_ard(seed: int = 42, problem=DSGSWSProblem) -> pd.DataFrame:
    """比较启用 ARD 和不启用 ARD 的 GP 模型表现，判断参数各向异性建模是否值得保留。"""

    problem = resolve_problem(problem)
    rows = []
    for ard in [True, False]:
        optimizer = SWSBayesianOptimizer(backend="mock_dsg", problem=problem, n_initial=20, n_iterations=2, batch_size=1, seed=seed, ard=ard)
        result = optimizer.run()
        rows.append({"ard": ard, "hypervolume": result["best_hypervolume"], "feasible_rate": result["feasible_rate"]})
    return pd.DataFrame(rows)


def compare_robust(seed: int = 42, problem=DSGSWSProblem) -> pd.DataFrame:
    """比较名义排序与鲁棒排序的结果差异，用于评估制造公差分析是否改变最终候选优先级。"""

    problem = resolve_problem(problem)
    optimizer = SWSBayesianOptimizer(backend="mock_dsg", problem=problem, n_initial=20, n_iterations=2, batch_size=1, seed=seed)
    optimizer.run()
    pareto = optimizer.export_pareto_dataframe()
    records = robust_candidate_ranking(
        pareto[optimizer.problem.param_names].to_numpy(),
        simulator=optimizer.simulator,
        seed=seed,
        problem=problem,
    )
    rows = []
    for record in records[: min(10, len(records))]:
        rows.append(
            {
                "nominal_neg_Kc": record.nominal[0],
                "robust_neg_Kc": record.expected[0],
                "nominal_feasible_rate": 1.0,
                "robust_feasible_rate": record.feasibility_rate,
            }
        )
    return pd.DataFrame(rows)
