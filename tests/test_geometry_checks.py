from sws_bo.problems.dsg_bwo_problem import DSGSWSProblem
from sws_bo.utils.geometry_checks import is_valid_design


def test_illegal_geometry_rejected():
    bad = DSGSWSProblem.reference_design.copy()
    bad[3] = bad[1] + 0.01
    assert not is_valid_design(bad)
