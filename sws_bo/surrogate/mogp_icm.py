"""Experimental ICM-style multi-task GP interface."""

from __future__ import annotations

import torch
from botorch.fit import fit_gpytorch_model
from botorch.models import MultiTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood


def build_icm_model(train_X: torch.Tensor, train_Y: torch.Tensor) -> MultiTaskGP:
    """Build an experimental multitask GP from dense outputs.

    The default project path uses independent GPs because they are more stable
    for 50-250 samples.  This ICM-style model is kept as an optional extension.
    """

    n, d = train_X.shape
    tasks = []
    values = []
    inputs = []
    for task_idx in range(train_Y.shape[-1]):
        task_column = torch.full((n, 1), task_idx, dtype=train_X.dtype, device=train_X.device)
        inputs.append(torch.cat([train_X, task_column], dim=-1))
        values.append(train_Y[:, task_idx : task_idx + 1])
        tasks.append(task_idx)
    mt_X = torch.cat(inputs, dim=0)
    mt_Y = torch.cat(values, dim=0)
    return MultiTaskGP(mt_X, mt_Y, task_feature=d)


def fit_icm_model(model: MultiTaskGP) -> MultiTaskGP:
    """Fit the multitask GP."""

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    return model
