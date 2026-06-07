"""本模块实现项目默认的 independent GP 代理模型方案，采用 Matern-5/2 ARD 内核并面向多目标与约束场景提供稳定的训练和预测接口。"""

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
    """定义带 Matern-5/2 ARD 内核的单输出 GP 模型，是项目默认独立 GP 方案的基础单元。"""

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
    """针对每个输出维度分别构建一个 GP，并把它们组织成独立多输出代理模型。"""

    models = [
        AnisotropicMaternSingleTaskGP(train_X, train_Y[:, idx : idx + 1], ard=ard)
        for idx in range(train_Y.shape[-1])
    ]
    return ModelListGP(*models)


def fit_gp_model(model: ModelListGP) -> ModelListGP:
    """用精确边际似然训练独立 GP 列表，使每个目标或约束维度都完成拟合。"""

    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    return model


def predict(model: ModelListGP, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """对给定输入点预测所有输出的后验均值与标准差，为采集函数和分析模块提供统一接口。"""

    model.eval()
    with torch.no_grad():
        posterior = model.posterior(X)
        mean = posterior.mean
        variance = posterior.variance.clamp_min(1e-12)
    return mean, variance.sqrt()


def get_ard_lengthscales(model: ModelListGP) -> dict[str, dict[str, float]]:
    """从训练好的 GP 中提取每个输出对应的 ARD 长度尺度，用于后续敏感性分析。"""

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
    """把已训练 GP 的状态字典保存到磁盘，便于后续复现或离线分析。"""

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
    """从磁盘读取 GP 状态并恢复模型，避免每次分析都重新训练。"""

    model = build_independent_model(train_X, train_Y, ard=ard)
    model.load_state_dict(torch.load(Path(path), map_location=train_X.device))
    model.eval()
    return model
