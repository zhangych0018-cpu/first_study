"""本模块实现 DSG 问题上的若干基线搜索策略，用于与主贝叶斯优化方法做公平对比。它更强调实验可比性，而不是单次运行的复杂度。"""

from __future__ import annotations

import numpy as np
import torch

from ..acquisition.constraints import probability_feasible
from ..problems.dsg_bwo_problem import DSGSWSProblem
from ..problems import resolve_problem
from ..surrogate.train import train_independent_gp
from .initial_design import generate_hybrid_design, generate_lhs_design


def _evaluate_designs(simulator, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = []
    success = []
    for point in points:
        result = simulator.run(point)
        success.append(result.success)
        raw.append([result.Kc_mean, result.vp_std, result.ohmic_loss_mean, result.S11_max])
    return np.asarray(raw, dtype=float), np.asarray(success, dtype=bool)


def random_search(simulator, n_samples: int, seed: int = 42, problem=DSGSWSProblem) -> dict:
    """在参数边界内执行随机搜索基线，用于给主 BO 方法提供最简单的性能参考。"""

    problem = resolve_problem(problem)
    rng = np.random.default_rng(seed)
    points = rng.uniform(problem.bounds[:, 0], problem.bounds[:, 1], size=(n_samples, problem.dim))
    raw, success = _evaluate_designs(simulator, points)
    return {"points": points, "raw": raw, "success": success}


def lhs_search(simulator, n_samples: int, seed: int = 42, problem=DSGSWSProblem) -> dict:
    """使用拉丁超立方采样执行非序贯搜索基线，观察单纯覆盖式采样在 DSG 问题上的表现。"""

    problem = resolve_problem(problem)
    design = generate_lhs_design(n_samples=n_samples, seed=seed, normalized=False, problem=problem)
    raw, success = _evaluate_designs(simulator, design.valid)
    return {"points": design.valid, "raw": raw, "success": success}


def surrogate_then_optimize(simulator, n_initial: int, n_candidates: int = 1024, seed: int = 42, problem=DSGSWSProblem) -> dict:
    """先用初始设计数据训练一次代理模型，再直接优化代理的后验均值，作为简化版代理优化基线。"""

    problem = resolve_problem(problem)
    initial = generate_hybrid_design(n_initial, seed=seed, normalized=False, problem=problem).valid
    raw, success = _evaluate_designs(simulator, initial)
    train_X = torch.tensor(problem.normalize(initial[success]), dtype=torch.double)
    train_Y = torch.tensor(raw[success], dtype=torch.double)
    model = train_independent_gp(train_X, train_Y, ard=True)
    rng = np.random.default_rng(seed + 1)
    pool = rng.uniform(problem.bounds[:, 0], problem.bounds[:, 1], size=(n_candidates, problem.dim))
    X_pool = torch.tensor(problem.normalize(pool), dtype=torch.double)
    posterior_mean = model.posterior(X_pool).mean.detach().cpu().numpy()
    score = posterior_mean[:, 0] - posterior_mean[:, 1] - posterior_mean[:, 2]
    best = pool[np.argmax(score)]
    result = simulator.run(best)
    return {"best_point": best, "best_result": result.to_dict()}


def weighted_sum_bo(
    simulator,
    n_initial: int,
    n_iterations: int,
    weights: np.ndarray | None = None,
    seed: int = 42,
    problem=DSGSWSProblem,
) -> dict:
    """采用加权和目标的简单 BO 基线，用于与真正的多目标 qNEHVI 主方法进行对比。"""

    problem = resolve_problem(problem)
    weights = np.asarray(weights if weights is not None else [0.6, 0.2, 0.2], dtype=float)
    design = generate_hybrid_design(n_initial, seed=seed, normalized=False, problem=problem).valid
    raw, success = _evaluate_designs(simulator, design)
    X = design[success]
    Y = raw[success]
    history = []
    rng = np.random.default_rng(seed + 7)
    for _ in range(n_iterations):
        train_X = torch.tensor(problem.normalize(X), dtype=torch.double)
        train_Y = torch.tensor(Y, dtype=torch.double)
        model = train_independent_gp(train_X, train_Y, ard=True)
        pool = rng.uniform(problem.bounds[:, 0], problem.bounds[:, 1], size=(256, problem.dim))
        X_pool = torch.tensor(problem.normalize(pool), dtype=torch.double)
        posterior = model.posterior(X_pool)
        mean = posterior.mean.detach().cpu().numpy()
        std = np.sqrt(np.maximum(posterior.variance.detach().cpu().numpy(), 1e-12))
        utility = weights[0] * mean[:, 0] - weights[1] * mean[:, 1] - weights[2] * mean[:, 2] + 0.05 * std[:, 0]
        pf = probability_feasible(model, X_pool, threshold=problem.s11_constraint_db).detach().cpu().numpy()
        utility = utility * pf
        point = pool[np.argmax(utility)]
        result = simulator.run(point)
        history.append(result.to_dict())
        if result.success:
            X = np.vstack([X, point])
            Y = np.vstack([Y, [result.Kc_mean, result.vp_std, result.ohmic_loss_mean, result.S11_max]])
    return {"points": X, "raw": Y, "history": history}
