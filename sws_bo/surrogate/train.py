"""本模块负责代理模型训练流程中的公共组织逻辑，例如准备训练数据、选择模型实现和封装常见训练步骤。它把脚本层与具体 GP 实现解耦。"""

from __future__ import annotations

import torch

from .independent_gp import build_independent_model, fit_gp_model


def train_independent_gp(train_X: torch.Tensor, train_Y: torch.Tensor, ard: bool = True):
    """构建并训练项目默认的独立 GP 代理模型，为脚本层提供简洁的训练入口。"""

    model = build_independent_model(train_X, train_Y, ard=ard)
    return fit_gp_model(model)
