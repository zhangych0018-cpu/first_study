"""Constrained qNEHVI acquisition helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.optim import optimize_acqf
try:
    from botorch.sampling.normal import SobolQMCNormalSampler
except ImportError:  # botorch 0.7
    from botorch.sampling.samplers import SobolQMCNormalSampler

from ..problems.dsg_bwo_problem import DSGSWSProblem
from ..problems import resolve_problem
from .constraints import probability_feasible


@dataclass
class AcquisitionRecommendation:
    """Candidate batch with normalized and physical views."""

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
    """Construct constrained qNEHVI for the first three objectives."""

    problem = resolve_problem(problem)
    ref = torch.tensor(ref_point or problem.acquisition_ref_point, dtype=train_X.dtype, device=train_X.device)
    objective = IdentityMCMultiOutputObjective(outcomes=[0, 1, 2])
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
    """Optimize the acquisition over [0,1]^d."""

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
    """Recommend candidate points and enforce a minimum feasibility probability."""

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
    """Transform raw outputs to qNEHVI maximization space."""

    return torch.stack(
        [
            Y_raw_objectives[:, 0],
            -Y_raw_objectives[:, 1],
            -Y_raw_objectives[:, 2],
        ],
        dim=-1,
    )
