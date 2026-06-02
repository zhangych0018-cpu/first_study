from sws_bo.problems.dsg_bwo_problem import DSGSWSProblem
from sws_bo.utils.mock_dsg_cst import MockDSGCSTSimulator


def test_mock_dsg_cst_fields_complete():
    simulator = MockDSGCSTSimulator(seed=5)
    result = simulator.run(DSGSWSProblem.reference_design)
    payload = result.to_dict()
    assert result.success
    assert "Kc_TM21_mean" in payload
    assert "mode_ratio" in payload
    assert "f_TM21_ghz" in payload
    assert payload["Kc_TM21_mean"] > 0


def test_mock_dsg_reference_design_physics():
    simulator = MockDSGCSTSimulator(seed=7)
    result = simulator.run(DSGSWSProblem.reference_design)
    payload = result.to_dict()
    assert 96.0 <= payload["f_TM21_ghz"] <= 104.0
    assert 5.0 <= payload["Kc_TM21_mean"] <= 11.0
    assert payload["mode_ratio"] > 0.5
