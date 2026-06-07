"""本模块实现带约束的 qNEHVI 候选推荐流程，包括参考点组织、约束处理和多起点优化。它是 DSG 多目标贝叶斯优化中生成新设计点的核心组件之一。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.objective import GenericMCMultiOutputObjective
from botorch.optim import optimize_acqf
try:
    from botorch.sampling.normal import SobolQMCNormalSampler
except ImportError:  # 兼容较旧的 botorch 0.7 导入路径
    from botorch.sampling.samplers import SobolQMCNormalSampler

from ..problems.dsg_bwo_problem import DSGSWSProblem
from ..problems import resolve_problem
from .constraints import probability_feasible


@dataclass
class AcquisitionRecommendation:
    """用于保存一批采集函数推荐候选点，同时保留归一化坐标与物理参数坐标，方便优化器和结果输出同时使用。"""

    normalized: np.ndarray
    physical: np.ndarray
    acquisition_value: np.ndarray
    feasibility_probability: np.ndarray


def build_qnehvi_acquisition(
    model,
    train_X: torch.Tensor,
    train_Y_raw: torch.Tensor,
    ref_point: np.ndarray | None = None,
    mc_samples: int = 128,
    problem=DSGSWSProblem,
):
    """构建面向前三个目标的带约束 qNEHVI 采集函数，并把约束维度纳入可行性判断。"""

    problem = resolve_problem(problem)
    ref_source = problem.acquisition_ref_point if ref_point is None else ref_point
    ref = torch.tensor(ref_source, dtype=train_X.dtype, device=train_X.device)
    objective = GenericMCMultiOutputObjective(lambda samples, X=None: _acq_objective_transform(samples))
    sampler = SobolQMCNormalSampler(num_samples=mc_samples)
    constraints = [lambda samples: samples[..., 3] - problem.s11_constraint_db]
    return qNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref.tolist(),
        X_baseline=train_X,
        sampler=sampler,
        objective=objective,
        constraints=constraints,
        prune_baseline=False,
        cache_root=False,
    )


def optimize_acquisition(
    acq_function,
    bounds: torch.Tensor,
    q: int = 1,
    num_restarts: int = 10,
    raw_samples: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """在归一化超立方体 `[0,1]^d` 上对采集函数做多起点优化，寻找一批新的候选设计点。"""

    return optimize_acqf(
        acq_function=acq_function,
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )


def recommend_candidates(
    model,
    train_X: torch.Tensor,
    train_Y_raw: torch.Tensor,
    q: int = 1,
    pf_min: float = 0.80,
    num_restarts: int = 10,
    raw_samples: int = 256,
    problem=DSGSWSProblem,
) -> AcquisitionRecommendation:
    """在优化采集函数后返回推荐候选点，并额外施加最小可行概率阈值，避免把明显不可行的点交给后端仿真。"""

    problem = resolve_problem(problem)
    bounds = torch.stack(
        [
            torch.zeros(train_X.shape[-1], dtype=train_X.dtype, device=train_X.device),
            torch.ones(train_X.shape[-1], dtype=train_X.dtype, device=train_X.device),
        ]
    )
    acq = build_qnehvi_acquisition(model, train_X, train_Y_raw, problem=problem)
    candidates, values = optimize_acquisition(acq, bounds, q=q, num_restarts=num_restarts, raw_samples=raw_samples)
    pf = probability_feasible(model, candidates, problem=problem)
    if torch.any(pf < pf_min):
        pool = torch.rand(max(512, 64 * q), train_X.shape[-1], dtype=train_X.dtype, device=train_X.device)
        pool_pf = probability_feasible(model, pool, problem=problem)
        feasible_pool = pool[pool_pf >= pf_min]
        if len(feasible_pool) >= q:
            candidates = feasible_pool[:q]
            values = torch.zeros(q, dtype=train_X.dtype, device=train_X.device)
            pf = pool_pf[pool_pf >= pf_min][:q]
    normalized = candidates.detach().cpu().numpy()
    physical = problem.unnormalize(normalized)
    return AcquisitionRecommendation(
        normalized=normalized,
        physical=np.asarray(physical, dtype=float),
        acquisition_value=values.detach().cpu().numpy(),
        feasibility_probability=pf.detach().cpu().numpy(),
    )


def _acq_objective_transform(Y_raw_objectives: torch.Tensor) -> torch.Tensor:
    """把原始输出空间变换到 qNEHVI 的最大化空间，使最小化目标能以统一的正向收益形式被采集函数处理。"""

    return torch.stack(
        [
            Y_raw_objectives[..., 0],
            -Y_raw_objectives[..., 1],
            -Y_raw_objectives[..., 2],
        ],
        dim=-1,
    )
