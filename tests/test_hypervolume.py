"""本测试模块用于验证多目标超体积计算在简单可验证样例上的正确性。"""

import numpy as np

def test_hypervolume_simple_example():
    from sws_bo.analysis.hypervolume import compute_hypervolume

    points = np.array([[1.0, 2.0], [2.0, 1.0]])
    ref = np.array([3.0, 3.0])
    hv = compute_hypervolume(points, ref_point=ref, maximize=False)
    assert abs(hv - 3.0) < 1e-8
