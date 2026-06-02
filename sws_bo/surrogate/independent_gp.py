"""Independent Gaussian-process surrogate models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from botorch.fit import fit_gpytorch_model
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood


class AnisotropicMaternSingleTaskGP(SingleTaskGP):
    """Single-task GP with a Matern-5/2 ARD kernel."""

    def __init__(self, train_X: torch.Tensor, train_Y: torch.Tensor, ard: bool = True) -> None:
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            input_transform=Normalize(d=train_X.shape[-1]),
            outcome_transform=Standardize(m=train_Y.shape[-1]),
        )
        ard_dims = train_X.shape[-1] if ard else None
        self.covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=ard_dims))


def build_independent_model(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    ard: bool = True,
) -> ModelListGP:
    """Build one GP per output column."""

    models = [
        AnisotropicMaternSingleTaskGP(train_X, train_Y[:, idx : idx + 1], ard=ard)
        for idx in range(train_Y.shape[-1])
    ]
    return ModelListGP(*models)


def fit_gp_model(model: ModelListGP) -> ModelListGP:
    """Fit the independent GP list with exact marginal log likelihood."""

    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    return model


def predict(model: ModelListGP, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict posterior mean and standard deviation for all outputs."""

    model.eval()
    with torch.no_grad():
        posterior = model.posterior(X)
        mean = posterior.mean
        variance = posterior.variance.clamp_min(1e-12)
    return mean, variance.sqrt()


def get_ard_lengthscales(model: ModelListGP) -> dict[str, dict[str, float]]:
    """Extract per-output ARD lengthscales for sensitivity analysis."""

    result: dict[str, dict[str, float]] = {}
    for idx, sub_model in enumerate(model.models):
        base_kernel = getattr(sub_model.covar_module, "base_kernel", None)
        if base_kernel is None:
            continue
        lengths = base_kernel.lengthscale.detach().cpu().view(-1).numpy()
        result[f"output_{idx}"] = {
            f"x{param_idx}": float(lengths[param_idx]) for param_idx in range(len(lengths))
        }
    return result


def save_model(model: ModelListGP, path: str | Path) -> Path:
    """Persist a fitted GP state dict."""

    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path_obj)
    return path_obj


def load_model(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    path: str | Path,
    ard: bool = True,
) -> ModelListGP:
    """Load a fitted GP model from disk."""

    model = build_independent_model(train_X, train_Y, ard=ard)
    model.load_state_dict(torch.load(Path(path), map_location=train_X.device))
    model.eval()
    return model
