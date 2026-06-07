"""本测试模块用于验证项目默认问题入口与底层 DSG 问题定义之间的一致性。"""

import numpy as np

from sws_bo.problems.dsg_bwo_problem import DSGSWSProblem


def test_normalize_unnormalize_roundtrip():
    x = DSGSWSProblem.reference_design
    x_norm = DSGSWSProblem.normalize(x)
    x_back = DSGSWSProblem.unnormalize(x_norm)
    assert np.allclose(x, x_back, atol=1e-10)
