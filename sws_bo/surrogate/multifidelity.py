"""Autoregressive multi-fidelity correction utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


@dataclass
class MultiFidelityOutputModel:
    """One-output autoregressive multifidelity model."""

    rho: float
    delta_gp: GaussianProcessRegressor | None
    delta_mean: float


@dataclass
class MultiFidelityModel:
    """Container for per-output multifidelity corrections."""

    output_models: dict[str, MultiFidelityOutputModel]
    low_X: np.ndarray
    low_Y: np.ndarray


def _nearest_low_prediction(low_X: np.ndarray, low_y: np.ndarray, query_X: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(low_X[:, None, :] - query_X[None, :, :], axis=-1)
    indices = np.argmin(distances, axis=0)
    return low_y[indices]


def fit_autoregressive_multifidelity(
    low_X: np.ndarray,
    low_Y: np.ndarray,
    high_X: np.ndarray,
    high_Y: np.ndarray,
    output_names: list[str],
) -> MultiFidelityModel:
    """Fit per-output autoregressive multifidelity corrections."""

    models: dict[str, MultiFidelityOutputModel] = {}
    low_X = np.asarray(low_X, dtype=float)
    low_Y = np.asarray(low_Y, dtype=float)
    high_X = np.asarray(high_X, dtype=float)
    high_Y = np.asarray(high_Y, dtype=float)
    for idx, name in enumerate(output_names):
        low_pred = _nearest_low_prediction(low_X, low_Y[:, idx], high_X)
        denom = float(np.dot(low_pred, low_pred) + 1e-12)
        rho = float(np.dot(high_Y[:, idx], low_pred) / denom)
        delta = high_Y[:, idx] - rho * low_pred
        delta_gp = None
        if len(high_X) >= 3:
            kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-6)
            delta_gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=0)
            delta_gp.fit(high_X, delta)
        models[name] = MultiFidelityOutputModel(rho=rho, delta_gp=delta_gp, delta_mean=float(delta.mean()))
    return MultiFidelityModel(output_models=models, low_X=low_X, low_Y=low_Y)


def predict_high_fidelity(model: MultiFidelityModel, X: np.ndarray, output_names: list[str]) -> np.ndarray:
    """Predict corrected high-fidelity outputs."""

    X = np.asarray(X, dtype=float)
    outputs = []
    for idx, name in enumerate(output_names):
        cfg = model.output_models[name]
        low_pred = _nearest_low_prediction(model.low_X, model.low_Y[:, idx], X)
        if cfg.delta_gp is not None:
            delta = cfg.delta_gp.predict(X)
        else:
            delta = np.full(len(X), cfg.delta_mean)
        outputs.append(cfg.rho * low_pred + delta)
    return np.column_stack(outputs)


def estimate_bias(model: MultiFidelityModel) -> dict[str, dict[str, float]]:
    """Summarize multiplicative and additive bias terms."""

    return {
        name: {
            "rho": cfg.rho,
            "delta_mean": cfg.delta_mean,
        }
        for name, cfg in model.output_models.items()
    }


def calibrate_with_measurements(
    low_X: np.ndarray,
    low_Y: np.ndarray,
    measurement_df,
    output_names: list[str],
) -> tuple[MultiFidelityModel, np.ndarray]:
    """Fit and apply multifidelity calibration from a measurement dataframe."""

    high_X = measurement_df[[c for c in measurement_df.columns if c.startswith("x_")]].to_numpy()
    high_Y = measurement_df[output_names].to_numpy()
    model = fit_autoregressive_multifidelity(low_X, low_Y, high_X, high_Y, output_names)
    corrected = predict_high_fidelity(model, high_X, output_names)
    return model, corrected
