"""Problem registry for the DSG slow-wave structure."""

from __future__ import annotations

from .dsg_bwo_problem import DSGSWSProblem


PROBLEM_REGISTRY: dict[str, type] = {
    "default": DSGSWSProblem,
    "dsg": DSGSWSProblem,
    "dsg_bwo": DSGSWSProblem,
    "dsg_w_band_bwo": DSGSWSProblem,
}


def resolve_problem(problem: str | type | None) -> type:
    """Resolve a problem identifier to a problem class."""

    if problem is None:
        return DSGSWSProblem
    if isinstance(problem, str):
        key = problem.strip().lower()
        if key not in PROBLEM_REGISTRY:
            raise KeyError(f"Unknown problem '{problem}'. Available: {sorted(PROBLEM_REGISTRY)}")
        return PROBLEM_REGISTRY[key]
    return problem


__all__ = ["DSGSWSProblem", "PROBLEM_REGISTRY", "resolve_problem"]
