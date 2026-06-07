"""本测试模块用于验证初始设计采样器生成的样本数量、范围和几何有效性约束。"""

import numpy as np

from sws_bo.optimization.initial_design import generate_hybrid_design, generate_lhs_design, generate_sobol_design
from sws_bo.problems.dsg_bwo_problem import DSGSWSProblem


def test_designs_within_bounds():
    for generator in [generate_lhs_design, generate_sobol_design, generate_hybrid_design]:
        result = generator(16, seed=3, normalized=False)
        assert result.valid.shape[1] == DSGSWSProblem.dim
        assert np.all(result.valid >= DSGSWSProblem.bounds[:, 0] - 1e-12)
        assert np.all(result.valid <= DSGSWSProblem.bounds[:, 1] + 1e-12)
