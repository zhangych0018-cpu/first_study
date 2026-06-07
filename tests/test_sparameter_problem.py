"""本测试模块验证 S 参数降参问题定义和采集函数目标变换，确保真实 CST 可导出数据能进入 BO 数学接口。"""

from __future__ import annotations

import numpy as np
import torch

from sws_bo.acquisition.constrained_qnehvi import _acq_objective_transform
from sws_bo.problems import DSGSParameterProblem, resolve_problem


def test_resolve_sparameter_problem_preserves_five_design_variables():
    problem = resolve_problem("dsg_sparameter")
    assert problem is DSGSParameterProblem
    assert problem.dim == 5
    assert problem.param_names == ["W", "P", "T", "G", "H"]
    assert np.all(problem.hypervolume_ref_point > np.array([1.0, 1.0, 1.0]))


def test_acquisition_objective_transform_maximizes_quality_and_minimizes_penalties():
    raw = torch.tensor([[[[-1.2, 0.4, 1.2, -15.0]]]], dtype=torch.double)
    transformed = _acq_objective_transform(raw)
    assert transformed.shape == raw.shape[:-1] + (3,)
    assert transformed[..., 0].item() == -1.2
    assert transformed[..., 1].item() == -0.4
    assert transformed[..., 2].item() == -1.2
