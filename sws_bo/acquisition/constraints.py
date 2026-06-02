"""Constraint utilities for S11 feasibility handling."""

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
    """Compute P(S11 <= threshold) from the GP posterior."""

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
    """Boolean mask for hard constraint satisfaction."""

    problem = resolve_problem(problem)
    threshold = problem.s11_constraint_db if threshold is None else threshold
    return np.asarray(Y_raw)[:, 3] <= threshold


def constrained_objective_mask(
    Y_raw: np.ndarray,
    success_mask: np.ndarray | None = None,
    threshold: float | None = None,
    problem=DSGSWSProblem,
) -> np.ndarray:
    """Mask for rows that are both successful and S11-feasible."""

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
    """Pareto mask among feasible points only."""

    feasible = constrained_objective_mask(Y_raw, success_mask=success_mask, threshold=threshold, problem=problem)
    mask = np.zeros(len(objective_values), dtype=bool)
    if feasible.any():
        feasible_obj = np.asarray(objective_values)[feasible]
        pareto = is_pareto_efficient(feasible_obj)
        mask[np.where(feasible)[0][pareto]] = True
    return mask
