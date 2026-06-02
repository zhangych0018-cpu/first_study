"""Initial experimental design generators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..geometry.dsg_sws import validate_dsg_geometry
from ..problems import resolve_problem


@dataclass
class DesignGenerationResult:
    """Design generator output with optional invalid points."""

    valid: np.ndarray
    invalid: np.ndarray


def _design_result(points: list[np.ndarray], invalid: list[np.ndarray], dim: int) -> DesignGenerationResult:
    valid_arr = np.vstack(points) if points else np.empty((0, dim))
    invalid_arr = np.vstack(invalid) if invalid else np.empty((0, dim))
    return DesignGenerationResult(valid=valid_arr, invalid=invalid_arr)


def _default_validator(point: np.ndarray, problem) -> bool:
    return bool(validate_dsg_geometry(point).is_valid and problem.validate_x(point))


def _default_repair(point: np.ndarray, problem) -> np.ndarray | None:
    candidate = np.clip(point, problem.bounds[:, 0], problem.bounds[:, 1])
    if _default_validator(candidate, problem):
        return candidate
    for blend in np.linspace(0.15, 0.55, 3):
        repaired = (1.0 - blend) * candidate + blend * problem.reference_design
        repaired = np.clip(repaired, problem.bounds[:, 0], problem.bounds[:, 1])
        if _default_validator(repaired, problem):
            return repaired
    return None


def _validate_or_repair(points: np.ndarray, problem, validator=None, repair_fn=None) -> DesignGenerationResult:
    valid, invalid = [], []
    for point in points:
        repaired = (repair_fn or _default_repair)(point, problem)
        if repaired is not None and (validator or _default_validator)(repaired, problem):
            valid.append(repaired)
        else:
            invalid.append(point)
    return _design_result(valid, invalid, problem.dim)


def generate_lhs_design(
    n_samples: int,
    seed: int = 42,
    normalized: bool = False,
    problem=None,
    validator=None,
    repair_fn=None,
) -> DesignGenerationResult:
    """Generate a Latin-hypercube design inside the physical search bounds."""

    problem = resolve_problem(problem)
    rng = np.random.default_rng(seed)
    sample = np.empty((n_samples, problem.dim), dtype=float)
    for dim in range(problem.dim):
        perm = rng.permutation(n_samples)
        jitter = rng.random(n_samples)
        sample[:, dim] = (perm + jitter) / n_samples
    physical = problem.unnormalize(sample)
    result = _validate_or_repair(np.asarray(physical, dtype=float), problem, validator=validator, repair_fn=repair_fn)
    if normalized:
        result.valid = np.asarray(problem.normalize(result.valid), dtype=float)
        result.invalid = np.asarray(problem.normalize(result.invalid), dtype=float) if len(result.invalid) else result.invalid
    return result


def generate_sobol_design(
    n_samples: int,
    seed: int = 42,
    normalized: bool = False,
    problem=None,
    validator=None,
    repair_fn=None,
) -> DesignGenerationResult:
    """Generate a Sobol design inside the physical search bounds."""

    problem = resolve_problem(problem)
    if n_samples <= 0:
        empty = np.empty((0, problem.dim), dtype=float)
        return DesignGenerationResult(valid=empty.copy(), invalid=empty.copy())
    engine = torch.quasirandom.SobolEngine(dimension=problem.dim, scramble=True, seed=seed)
    sample = engine.draw(n_samples).detach().cpu().numpy()
    physical = problem.unnormalize(sample)
    result = _validate_or_repair(np.asarray(physical, dtype=float), problem, validator=validator, repair_fn=repair_fn)
    if normalized:
        result.valid = np.asarray(problem.normalize(result.valid), dtype=float)
        result.invalid = np.asarray(problem.normalize(result.invalid), dtype=float) if len(result.invalid) else result.invalid
    return result


def generate_local_perturbation_design(
    n_samples: int,
    seed: int = 42,
    normalized: bool = False,
    scale: float = 0.08,
    problem=None,
    validator=None,
    repair_fn=None,
) -> DesignGenerationResult:
    """Perturb the reference design locally."""

    problem = resolve_problem(problem)
    rng = np.random.default_rng(seed)
    span = problem.bounds[:, 1] - problem.bounds[:, 0]
    points = []
    for _ in range(n_samples):
        perturbed = problem.reference_design + rng.normal(0.0, scale * span)
        points.append(np.clip(perturbed, problem.bounds[:, 0], problem.bounds[:, 1]))
    result = _validate_or_repair(np.vstack(points), problem, validator=validator, repair_fn=repair_fn)
    if normalized:
        result.valid = np.asarray(problem.normalize(result.valid), dtype=float)
        result.invalid = np.asarray(problem.normalize(result.invalid), dtype=float) if len(result.invalid) else result.invalid
    return result


def generate_hybrid_design(
    n_samples: int,
    seed: int = 42,
    normalized: bool = False,
    lhs_fraction: float = 0.5,
    sobol_fraction: float = 0.3,
    problem=None,
    validator=None,
    repair_fn=None,
) -> DesignGenerationResult:
    """Combine LHS, Sobol and local reference perturbations."""

    problem = resolve_problem(problem)
    n_lhs = int(round(n_samples * lhs_fraction))
    n_sobol = int(round(n_samples * sobol_fraction))
    n_local = max(0, n_samples - n_lhs - n_sobol)
    lhs = generate_lhs_design(n_lhs, seed=seed, normalized=False, problem=problem, validator=validator, repair_fn=repair_fn)
    sobol = generate_sobol_design(n_sobol, seed=seed + 1, normalized=False, problem=problem, validator=validator, repair_fn=repair_fn)
    local = generate_local_perturbation_design(n_local, seed=seed + 2, normalized=False, problem=problem, validator=validator, repair_fn=repair_fn)
    valid = np.vstack([arr for arr in [lhs.valid, sobol.valid, local.valid] if len(arr)]) if any(
        len(arr) for arr in [lhs.valid, sobol.valid, local.valid]
    ) else np.empty((0, problem.dim))
    invalid = np.vstack([arr for arr in [lhs.invalid, sobol.invalid, local.invalid] if len(arr)]) if any(
        len(arr) for arr in [lhs.invalid, sobol.invalid, local.invalid]
    ) else np.empty((0, problem.dim))
    if len(valid) > n_samples:
        valid = valid[:n_samples]
    result = DesignGenerationResult(valid=valid, invalid=invalid)
    if normalized:
        result.valid = np.asarray(problem.normalize(result.valid), dtype=float)
        result.invalid = np.asarray(problem.normalize(result.invalid), dtype=float) if len(result.invalid) else result.invalid
    return result
