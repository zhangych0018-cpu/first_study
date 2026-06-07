"""本测试模块用于验证 Pareto 有效性判断和可行前沿提取函数在小型样例上的表现。"""

import numpy as np

from sws_bo.analysis.pareto import feasible_pareto_front, is_pareto_efficient


def test_pareto_mask_correct():
    obj = np.array([[1.0, 2.0], [1.5, 1.5], [2.0, 2.0], [0.9, 2.5]])
    mask = is_pareto_efficient(obj)
    assert mask.tolist() == [True, True, False, True]


def test_feasible_pareto_correct():
    obj = np.array([[1.0, 2.0], [1.5, 1.5], [2.0, 2.0], [0.9, 2.5]])
    feasible = np.array([True, False, True, True])
    mask = feasible_pareto_front(obj, feasible)
    assert mask.tolist() == [True, False, False, True]
