"""本模块提供实验性的 ICM / MultiTaskGP 扩展接口，用于在需要时探索输出相关性的联合建模方案。默认主流程并不强依赖它。"""

from __future__ import annotations

import torch
from botorch.fit import fit_gpytorch_model
from botorch.models import MultiTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood


def build_icm_model(train_X: torch.Tensor, train_Y: torch.Tensor) -> MultiTaskGP:
    """构建实验性的多任务 GP / ICM 模型，用于探索输出相关性联合建模的可能收益。"""

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
    """训练实验性多任务 GP 模型，为后续对比独立 GP 与联合 GP 的差异做准备。"""

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    return model
