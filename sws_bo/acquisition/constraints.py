"""本模块集中处理可行概率、硬筛选与可行 Pareto 掩码等约束逻辑，确保约束语义在代理建模、采集函数和分析阶段保持一致。"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import norm

from ..analysis.pareto import is_pareto_efficient
from ..problems.dsg_bwo_problem import DSGSWSProblem
from ..problems import resolve_problem
from ..surrogate.independent_gp import predict


def probability_feasible(
    model,
    X: torch.Tensor,
    constraint_index: int = 3,
    threshold: float | None = None,
    problem=DSGSWSProblem,
) -> torch.Tensor:
    """基于 GP 后验分布计算 `S11` 满足阈值约束的概率，用于软约束筛选与采集函数加权。"""

    problem = resolve_problem(problem)
    threshold = problem.s11_constraint_db if threshold is None else threshold
    mean, std = predict(model, X)
    mean_np = mean[..., constraint_index].detach().cpu().numpy()
    std_np = std[..., constraint_index].detach().cpu().numpy()
    std_np = np.maximum(std_np, 1e-9)
    pf = norm.cdf((threshold - mean_np) / std_np)
    return torch.tensor(pf, dtype=X.dtype, device=X.device)


def hard_feasibility_filter(
    Y_raw: np.ndarray,
    threshold: float | None = None,
    problem=DSGSWSProblem,
) -> np.ndarray:
    """根据硬阈值直接生成布尔可行掩码，适合在结果统计和后处理阶段快速筛掉不满足约束的样本。"""

    problem = resolve_problem(problem)
    threshold = problem.s11_constraint_db if threshold is None else threshold
    return np.asarray(Y_raw)[:, 3] <= threshold


def constrained_objective_mask(
    Y_raw: np.ndarray,
    success_mask: np.ndarray | None = None,
    threshold: float | None = None,
    problem=DSGSWSProblem,
) -> np.ndarray:
    """生成同时满足“仿真成功”和 `S11` 约束的样本掩码，作为下游 Pareto 和训练筛选的基础。"""

    feasible = hard_feasibility_filter(Y_raw, threshold=threshold, problem=problem)
    if success_mask is None:
        return feasible
    return feasible & np.asarray(success_mask, dtype=bool)


def feasible_pareto_mask(
    objective_values: np.ndarray,
    Y_raw: np.ndarray,
    success_mask: np.ndarray | None = None,
    threshold: float | None = None,
    problem=DSGSWSProblem,
) -> np.ndarray:
    """仅在可行样本内部计算 Pareto 掩码，避免不可行点干扰多目标前沿判断。"""

    feasible = constrained_objective_mask(Y_raw, success_mask=success_mask, threshold=threshold, problem=problem)
    mask = np.zeros(len(objective_values), dtype=bool)
    if feasible.any():
        feasible_obj = np.asarray(objective_values)[feasible]
        pareto = is_pareto_efficient(feasible_obj)
        mask[np.where(feasible)[0][pareto]] = True
    return mask
