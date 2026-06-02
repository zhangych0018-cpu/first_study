"""Problem definition for the W-band DSG slow-wave structure."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch


class DSGSWSProblem:
    """Five-dimensional DSG slow-wave structure inverse-design problem."""

    name = "DSG_W_Band_BWO"
    dim = 5
    param_names = ["W", "P", "T", "G", "H"]
    param_bounds = {
        "W": (2.8, 3.6),
        "P": (0.70, 0.90),
        "T": (0.22, 0.40),
        "G": (0.12, 0.30),
        "H": (0.45, 0.75),
    }
    bounds = np.array(
        [
            (2.8, 3.6),
            (0.70, 0.90),
            (0.22, 0.40),
            (0.12, 0.30),
            (0.45, 0.75),
        ],
        dtype=float,
    )
    reference_design = np.array([3.2, 0.8, 0.3, 0.2, 0.6], dtype=float)
    target_names = ["Kc_mean", "vp_std", "ohmic_loss_mean", "S11_max"]
    objective_names = ["neg_Kc_mean", "vp_std", "ohmic_loss_mean"]
    num_objectives = 3
    num_outputs = 4
    s11_constraint_db = -10.0
    mode_ratio_min = 1.2
    target_frequency_ghz = 100.0
    working_band_ghz = (96.0, 110.0)
    hypervolume_ref_point = np.array([-3.0, 0.15, 0.22], dtype=float)
    acquisition_ref_point = np.array([2.0, 0.12, 0.18], dtype=float)
    default_tolerance_spec = {
        "W": {"type": "gaussian", "scale": 0.010},
        "P": {"type": "gaussian", "scale": 0.004},
        "T": {"type": "gaussian", "scale": 0.003},
        "G": {"type": "gaussian", "scale": 0.003},
        "H": {"type": "gaussian", "scale": 0.005},
    }

    @classmethod
    def normalize(cls, x: np.ndarray | torch.Tensor | Iterable[float]) -> np.ndarray | torch.Tensor:
        """Normalize physical parameters to [0, 1]^5."""

        if isinstance(x, torch.Tensor):
            lower = torch.tensor(cls.bounds[:, 0], dtype=x.dtype, device=x.device)
            upper = torch.tensor(cls.bounds[:, 1], dtype=x.dtype, device=x.device)
            return (x - lower) / (upper - lower)
        arr = np.asarray(x, dtype=float)
        return (arr - cls.bounds[:, 0]) / (cls.bounds[:, 1] - cls.bounds[:, 0])

    @classmethod
    def unnormalize(cls, x_norm: np.ndarray | torch.Tensor | Iterable[float]) -> np.ndarray | torch.Tensor:
        """Map normalized parameters back to physical units."""

        if isinstance(x_norm, torch.Tensor):
            lower = torch.tensor(cls.bounds[:, 0], dtype=x_norm.dtype, device=x_norm.device)
            upper = torch.tensor(cls.bounds[:, 1], dtype=x_norm.dtype, device=x_norm.device)
            return lower + x_norm * (upper - lower)
        arr = np.asarray(x_norm, dtype=float)
        return cls.bounds[:, 0] + arr * (cls.bounds[:, 1] - cls.bounds[:, 0])

    @classmethod
    def validate_x(cls, x: np.ndarray | torch.Tensor | Iterable[float]) -> bool:
        """Validate dimensionality and bound membership."""

        arr = np.asarray(x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x, dtype=float)
        if arr.shape[-1] != cls.dim:
            return False
        return bool(np.all(arr >= cls.bounds[:, 0]) and np.all(arr <= cls.bounds[:, 1]))

    @classmethod
    def get_bounds(cls, normalized: bool = True) -> np.ndarray:
        """Return either normalized or physical bounds."""

        if normalized:
            return np.tile(np.array([[0.0, 1.0]], dtype=float), (cls.dim, 1))
        return cls.bounds.copy()

    @classmethod
    def objective_transform(cls, raw_result: dict[str, float] | np.ndarray | torch.Tensor) -> np.ndarray:
        """Transform DSG outputs into the canonical BO objective space."""

        if isinstance(raw_result, dict):
            kc = raw_result.get("Kc_TM21_mean", raw_result.get("Kc_mean"))
            vp_std = raw_result.get("sync_error", raw_result.get("vp_TM21_std", raw_result.get("vp_std")))
            loss = raw_result["ohmic_loss_mean"]
            s11 = raw_result["S11_max"]
            return np.array([-kc, vp_std, loss, s11], dtype=float)
        arr = np.asarray(raw_result, dtype=float)
        return np.stack([-arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]], axis=-1)

    @classmethod
    def check_constraint(cls, y_constraint: float | np.ndarray | torch.Tensor) -> bool:
        """Check the S11 hard constraint for legacy compatibility."""

        value = float(np.asarray(y_constraint).reshape(-1)[0])
        return value <= cls.s11_constraint_db

    @classmethod
    def check_constraints(cls, raw_result: dict[str, float] | np.ndarray | torch.Tensor) -> bool:
        """Check S11 and mode-ratio constraints."""

        if isinstance(raw_result, dict):
            s11_ok = float(raw_result["S11_max"]) <= cls.s11_constraint_db
            mode_ratio = float(raw_result.get("mode_ratio", cls.mode_ratio_min))
            return bool(s11_ok and mode_ratio >= cls.mode_ratio_min)
        arr = np.asarray(raw_result, dtype=float).reshape(-1)
        s11_ok = float(arr[3]) <= cls.s11_constraint_db
        mode_ratio = float(arr[4]) if arr.size > 4 else cls.mode_ratio_min
        return bool(s11_ok and mode_ratio >= cls.mode_ratio_min)

    @classmethod
    def as_dict(cls, x: np.ndarray | Iterable[float]) -> dict[str, float]:
        """Convert array-like data into a named parameter dictionary."""

        arr = np.asarray(x, dtype=float)
        return {name: float(arr[idx]) for idx, name in enumerate(cls.param_names)}
