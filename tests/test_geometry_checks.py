"""本测试模块用于验证工程代理几何检查函数在合法与非法设计上的判定行为。"""

from sws_bo.problems.dsg_bwo_problem import DSGSWSProblem
from sws_bo.utils.geometry_checks import is_valid_design


def test_illegal_geometry_rejected():
    bad = DSGSWSProblem.reference_design.copy()
    bad[3] = bad[1] + 0.01
    assert not is_valid_design(bad)
