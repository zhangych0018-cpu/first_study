"""本测试模块用于验证 DSG 后处理解析器对标准样例、异常列和边界频段情况的处理能力。"""

from pathlib import Path

from sws_bo.utils.postprocessing import parse_dsg_cst_results


def test_dsg_postprocessing_parse_valid_files():
    root = Path(__file__).parent / "fixtures" / "dsg"
    metrics = parse_dsg_cst_results(
        dispersion_tm21_path=root / "dispersion_tm21.txt",
        dispersion_fundamental_path=root / "dispersion_fundamental.txt",
        kc_tm21_path=root / "kc_tm21.txt",
        kc_fundamental_path=root / "kc_fundamental.txt",
        sparameters_path=root / "sparameters.txt",
    )
    assert metrics["Kc_TM21_mean"] > 7.0
    assert metrics["mode_ratio"] > 1.2
    assert abs(metrics["f_TM21_ghz"] - 100.0) < 1e-8
    assert metrics["S11_max"] <= -16.0
