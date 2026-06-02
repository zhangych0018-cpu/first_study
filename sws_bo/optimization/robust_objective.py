"""Manufacturing-tolerance-aware robust objective utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..problems.dsg_bwo_problem import DSGSWSProblem
from ..problems import resolve_problem


DEFAULT_TOLERANCE_SPEC_MM = {
    "W": {"type": "gaussian", "scale": 0.010},
    "P": {"type": "gaussian", "scale": 0.004},
    "T": {"type": "gaussian", "scale": 0.003},
    "G": {"type": "gaussian", "scale": 0.003},
    "H": {"type": "gaussian", "scale": 0.005},
}


@dataclass
class RobustRankingRecord:
    """Nominal and robust summary for one candidate."""

    design: np.ndarray
    nominal: np.ndarray
    expected: np.ndarray
    worst_case: np.ndarray
    cvar: np.ndarray
    feasibility_rate: float
    robust_score: float


def sample_manufacturing_perturbations(
    x: np.ndarray,
    tolerance_spec: dict | None = None,
    n_samples: int = 32,
    seed: int = 0,
    problem=DSGSWSProblem,
) -> np.ndarray:
    """Sample manufacturing perturbations in physical units."""

    problem = resolve_problem(problem)
    rng = np.random.default_rng(seed)
    spec = tolerance_spec or getattr(problem, "default_tolerance_spec", DEFAULT_TOLERANCE_SPEC_MM)
    x = np.asarray(x, dtype=float)
    samples = np.tile(x, (n_samples, 1))
    for idx, name in enumerate(problem.param_names):
        cfg = spec[name]
        if cfg["type"] == "uniform":
            noise = rng.uniform(-cfg["scale"], cfg["scale"], size=n_samples)
        else:
            noise = rng.normal(0.0, cfg["scale"], size=n_samples)
        samples[:, idx] += noise
    return np.clip(samples, problem.bounds[:, 0], problem.bounds[:, 1])


def evaluate_expected_objective(objective_values: np.ndarray) -> np.ndarray:
    """Expected minimization objective under tolerance samples."""

    return np.asarray(objective_values, dtype=float).mean(axis=0)


def evaluate_worst_case_objective(objective_values: np.ndarray) -> np.ndarray:
    """Worst per-objective value across tolerance samples."""

    return np.asarray(objective_values, dtype=float).max(axis=0)


def evaluate_cvar_objective(objective_values: np.ndarray, alpha: float = 0.95) -> np.ndarray:
    """CVaR on the upper tail for minimization objectives."""

    values = np.asarray(objective_values, dtype=float)
    k = max(1, int(np.ceil(alpha * len(values))))
    sorted_values = np.sort(values, axis=0)
    return sorted_values[k - 1 :].mean(axis=0)


def _surrogate_evaluate(surrogate_model, samples: np.ndarray, problem=DSGSWSProblem) -> np.ndarray:
    problem = resolve_problem(problem)
    X = torch.tensor(problem.normalize(samples), dtype=torch.double)
    posterior = surrogate_model.posterior(X)
    return posterior.mean.detach().cpu().numpy()


def robust_candidate_ranking(
    candidates: np.ndarray,
    simulator=None,
    surrogate_model=None,
    tolerance_spec: dict | None = None,
    n_samples: int = 32,
    seed: int = 0,
    problem=DSGSWSProblem,
) -> list[RobustRankingRecord]:
    """Rank candidates by tolerance-aware performance."""

    problem = resolve_problem(problem)
    if simulator is None and surrogate_model is None:
        raise ValueError("Either simulator or surrogate_model must be provided.")

    records: list[RobustRankingRecord] = []
    for idx, candidate in enumerate(np.asarray(candidates, dtype=float)):
        perturbations = sample_manufacturing_perturbations(candidate, tolerance_spec, n_samples, seed + idx, problem=problem)
        if simulator is not None:
            raw = np.array(
                [
                    [res.Kc_mean, res.vp_std, res.ohmic_loss_mean, res.S11_max]
                    for res in [simulator.run(x) for x in perturbations]
                    if res.success
                ],
                dtype=float,
            )
        else:
            raw = _surrogate_evaluate(surrogate_model, perturbations, problem=problem)
        if len(raw) == 0:
            continue
        obj = problem.objective_transform(raw)
        feasible_rate = float(np.mean(raw[:, 3] <= problem.s11_constraint_db))
        expected = evaluate_expected_objective(obj[:, :3])
        worst = evaluate_worst_case_objective(obj[:, :3])
        cvar = evaluate_cvar_objective(obj[:, :3])
        nominal_raw = raw[0]
        nominal = problem.objective_transform(nominal_raw)[0, :3] if nominal_raw.ndim > 1 else problem.objective_transform(nominal_raw)[:3]
        robust_score = float(-(expected[0] + expected[1] + expected[2]) + 0.1 * feasible_rate)
        records.append(
            RobustRankingRecord(
                design=candidate,
                nominal=np.asarray(nominal, dtype=float),
                expected=expected,
                worst_case=worst,
                cvar=cvar,
                feasibility_rate=feasible_rate,
                robust_score=robust_score,
            )
        )
    records.sort(key=lambda record: record.robust_score, reverse=True)
    return records
