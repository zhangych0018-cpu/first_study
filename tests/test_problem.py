import numpy as np

from sws_bo.problems.dsg_bwo_problem import DSGSWSProblem


def test_normalize_unnormalize_roundtrip():
    x = DSGSWSProblem.reference_design
    x_norm = DSGSWSProblem.normalize(x)
    x_back = DSGSWSProblem.unnormalize(x_norm)
    assert np.allclose(x, x_back, atol=1e-10)
