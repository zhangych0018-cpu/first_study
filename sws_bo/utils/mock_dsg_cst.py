"""Mock CST-like simulator for the W-band DSG slow-wave structure."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np
import torch

from ..data_schema import SimulationResult
from ..forward_model import ForwardSimulator
from ..geometry.dsg_sws import validate_dsg_geometry
from ..problems.dsg_bwo_problem import DSGSWSProblem


@dataclass
class MockDSGConfig:
    """Configuration for the analytic DSG mock simulator."""

    seed: int = 42
    noise_scale: float = 1.0
    deterministic: bool = True


class MockDSGCSTSimulator(ForwardSimulator):
    """Analytic DSG response model that mimics W-band cold-structure tradeoffs."""

    def __init__(self, seed: int = 42, noise_scale: float = 1.0, deterministic: bool = True) -> None:
        self.config = MockDSGConfig(seed=seed, noise_scale=noise_scale, deterministic=deterministic)

    def run(self, x: dict | np.ndarray | torch.Tensor) -> SimulationResult:
        """Evaluate one DSG design point."""

        start = time.perf_counter()
        x_phys = self._to_physical_array(x)
        validation = validate_dsg_geometry(x_phys)
        if not validation.is_valid:
            return SimulationResult(
                Kc_mean=np.nan,
                vp_std=np.nan,
                ohmic_loss_mean=np.nan,
                S11_max=np.nan,
                sim_time=time.perf_counter() - start,
                success=False,
                failure_reason="invalid_dsg_geometry",
                extra_outputs={"mode_ratio": np.nan, "f_TM21_ghz": np.nan, "f_fund_ghz": np.nan, "Kc_TM21_mean": np.nan, "sync_error": np.nan, "vp_TM21_std": np.nan},
            )

        rng = self._make_rng(x_phys)
        z = DSGSWSProblem.normalize(x_phys)
        ref = DSGSWSProblem.normalize(DSGSWSProblem.reference_design)
        q = (z - ref) / 0.45
        width, period, tunnel, thickness, height = q

        resonance = np.exp(
            -0.5 * (
                1.5 * (period - 0.10) ** 2
                + 0.9 * (tunnel + 0.08) ** 2
                + 1.0 * (thickness - 0.05) ** 2
                + 0.8 * (height - 0.04) ** 2
                + 0.7 * (width - 0.02) ** 2
            )
        )
        mode_mixing = np.sin(2.6 * np.pi * z[1]) * np.cos(1.7 * np.pi * z[4]) + 0.45 * np.sin(1.8 * np.pi * z[0])
        competition_penalty = 0.18 * abs(width + 0.3 * height) + 0.22 * abs(period - 0.4 * tunnel)
        instability = max(0.0, 0.9 * abs(height / max(x_phys[2], 1e-9) - 2.15) + 0.4 * abs(thickness + 0.25 * period) - 0.95)

        if rng.uniform() < min(0.10, instability * 0.10):
            return SimulationResult(
                Kc_mean=np.nan,
                vp_std=np.nan,
                ohmic_loss_mean=np.nan,
                S11_max=np.nan,
                sim_time=time.perf_counter() - start,
                success=False,
                failure_reason="mock_dsg_solver_instability",
                extra_outputs={"mode_ratio": np.nan, "f_TM21_ghz": np.nan, "f_fund_ghz": np.nan, "Kc_TM21_mean": np.nan, "sync_error": np.nan, "vp_TM21_std": np.nan},
            )

        f_tm21 = (
            100.0
            - 19.0 * period
            + 5.2 * tunnel
            - 2.8 * height
            + 1.9 * thickness
            + 2.0 * width
            + 1.4 * np.sin(2.0 * np.pi * z[1])
            - 0.9 * np.cos(1.5 * np.pi * z[3] * z[4])
        )
        f_tm21 += rng.normal(0.0, 0.22 * self.config.noise_scale)

        f_gap = 7.0 + 4.2 * (z[0] - 0.5) - 1.8 * abs(period) + 0.9 * height - 0.7 * thickness
        f_fund = f_tm21 - f_gap + rng.normal(0.0, 0.12 * self.config.noise_scale)

        kc_tm21 = (
            5.8
            + 2.6 * resonance
            + 0.75 * mode_mixing
            + 0.65 * np.cos(1.9 * np.pi * z[2])
            - 1.2 * abs(tunnel)
            + 0.55 * (height - thickness)
        )
        kc_tm21 += rng.normal(0.0, 0.10 * self.config.noise_scale)
        kc_tm21 = float(np.clip(kc_tm21, 2.2, 12.5))

        vp_tm21_std = (
            0.007
            + 0.012 * (period - 0.1) ** 2
            + 0.004 * (height - 0.1) ** 2
            + 0.003 * abs(thickness)
            + 0.0015 * (1.0 - resonance)
        )
        vp_tm21_std += rng.normal(0.0, 0.0006 * self.config.noise_scale)
        vp_tm21_std = float(np.clip(vp_tm21_std, 0.004, 0.060))

        sync_error = 0.010 + abs(f_tm21 - DSGSWSProblem.target_frequency_ghz) / 42.0 + 0.30 * vp_tm21_std
        sync_error += 0.012 * abs(period) + rng.normal(0.0, 0.0010 * self.config.noise_scale)
        sync_error = float(np.clip(sync_error, 0.004, 0.18))

        mode_ratio = 1.35 + 0.35 * (z[0] - 0.5) + 0.20 * resonance - 0.55 * competition_penalty
        mode_ratio += 0.10 * np.sin(2.2 * np.pi * z[3]) + rng.normal(0.0, 0.03 * self.config.noise_scale)
        mode_ratio = float(np.clip(mode_ratio, 0.55, 2.20))

        ohmic_loss = (
            0.065
            + 0.012 * kc_tm21 / 8.0
            + 0.028 * abs(height)
            + 0.018 * abs(thickness)
            + 0.015 * max(0.0, 0.26 - x_phys[2]) / 0.08
            + 0.008 * (1.0 - resonance)
        )
        ohmic_loss += rng.normal(0.0, 0.0022 * self.config.noise_scale)
        ohmic_loss = float(np.clip(ohmic_loss, 0.040, 0.220))

        s11 = (
            -17.5
            + 8.8 * (0.45 * period - 0.22 * width + 0.30 * thickness + 0.36 * height) ** 2
            + 2.6 * max(0.0, 1.15 - mode_ratio)
            - 2.1 * resonance
        )
        s11 += rng.normal(0.0, 0.35 * self.config.noise_scale)
        s11 = float(np.clip(s11, -28.0, -4.5))

        s21 = float(np.clip(-(ohmic_loss * 18.0) + rng.normal(0.0, 0.10), -5.5, -0.3))
        bandwidth = float(np.clip(14.0 - 55.0 * sync_error - 2.0 * max(0.0, 1.2 - mode_ratio), 1.0, 11.0))

        return SimulationResult(
            Kc_mean=kc_tm21,
            vp_std=sync_error,
            ohmic_loss_mean=ohmic_loss,
            S11_max=s11,
            S21_mean=s21,
            bandwidth_estimate=bandwidth,
            sim_time=time.perf_counter() - start,
            success=True,
            metadata={"backend": "mock_dsg", "geometry_warnings": validation.warnings},
            extra_outputs={
                "Kc_TM21_mean": kc_tm21,
                "sync_error": sync_error,
                "vp_TM21_std": vp_tm21_std,
                "mode_ratio": mode_ratio,
                "f_TM21_ghz": float(f_tm21),
                "f_fund_ghz": float(f_fund),
            },
        )

    def _to_physical_array(self, x: dict | np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(x, dict):
            return np.array([x[name] for name in DSGSWSProblem.param_names], dtype=float)
        if isinstance(x, torch.Tensor):
            arr = x.detach().cpu().numpy()
        else:
            arr = np.asarray(x, dtype=float)
        if np.all((arr >= 0.0) & (arr <= 1.0)):
            arr = DSGSWSProblem.unnormalize(arr)
        return np.asarray(arr, dtype=float)

    def _make_rng(self, x_phys: np.ndarray) -> np.random.Generator:
        if not self.config.deterministic:
            return np.random.default_rng()
        digest = hashlib.sha256(np.round(x_phys, 6).tobytes()).hexdigest()
        offset = int(digest[:8], 16)
        return np.random.default_rng(self.config.seed + offset)
