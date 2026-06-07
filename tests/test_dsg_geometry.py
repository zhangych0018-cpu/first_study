"""本测试模块用于验证 DSG 几何配置、边界规则和告警逻辑，确保结构参数转几何描述时不会出现明显错误。"""

from sws_bo.geometry.dsg_sws import DSGGeometryBuilder, validate_dsg_geometry
from sws_bo.problems.dsg_bwo_problem import DSGSWSProblem


def test_dsg_geometry_reference_valid():
    validation = validate_dsg_geometry(DSGSWSProblem.reference_design)
    assert validation.is_valid


def test_dsg_geometry_requires_g_less_than_p():
    bad = DSGSWSProblem.reference_design.copy()
    bad[3] = bad[1] + 0.01
    validation = validate_dsg_geometry(bad)
    assert not validation.is_valid


def test_dsg_geometry_builder_exports_cst_parameters():
    builder = DSGGeometryBuilder()
    payload = builder.export_cst_parameter_dict(DSGSWSProblem.reference_design)
    assert payload["W"] == DSGSWSProblem.reference_design[0]
    assert payload["N_periods"] == 80
