"""Typed data containers used across the project."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SimulationResult:
    """Canonical output from any forward simulator backend."""

    Kc_mean: float
    vp_std: float
    ohmic_loss_mean: float
    S11_max: float
    S21_mean: float | None = None
    bandwidth_estimate: float | None = None
    sim_time: float = 0.0
    success: bool = True
    failure_reason: str | None = None
    raw_files: dict[str, str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    extra_outputs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a serializable dictionary."""

        payload = asdict(self)
        extra = payload.pop("extra_outputs", {}) or {}
        payload.update(extra)
        return payload


@dataclass
class CandidateRecommendation:
    """Acquisition recommendation with both normalized and physical forms."""

    normalized_x: list[float]
    physical_x: list[float]
    score: float
    feasibility_probability: float | None = None


@dataclass
class RobustEvaluation:
    """Robust objective summary under manufacturing perturbations."""

    nominal_objective: list[float]
    expected_objective: list[float]
    worst_case_objective: list[float]
    cvar_objective: list[float]
    feasibility_rate: float
