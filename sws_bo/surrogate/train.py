"""Training helpers for surrogate models."""

from __future__ import annotations

import torch

from .independent_gp import build_independent_model, fit_gp_model


def train_independent_gp(train_X: torch.Tensor, train_Y: torch.Tensor, ard: bool = True):
    """Build and fit the default independent GP surrogate."""

    model = build_independent_model(train_X, train_Y, ard=ard)
    return fit_gp_model(model)
