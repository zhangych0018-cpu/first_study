from pathlib import Path

import pytest

from sws_bo.utils.postprocessing import (
    PostprocessingEmptyBandError,
    PostprocessingFileMissingError,
    PostprocessingFormatError,
    parse_dsg_cst_results,
    parse_dsg_dispersion,
)


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _bundle_paths(bundle_name: str) -> dict[str, Path]:
    root = FIXTURES_DIR / bundle_name
    return {
        "dispersion_tm21_path": root / "dispersion_tm21.txt",
        "dispersion_fundamental_path": root / "dispersion_fundamental.txt",
        "kc_tm21_path": root / "kc_tm21.txt",
        "kc_fundamental_path": root / "kc_fundamental.txt",
        "sparameters_path": root / "sparameters.txt",
    }


@pytest.mark.parametrize("bundle_name", ["dsg", "dsg_no_header"])
def test_postprocessing_parse_valid_files(bundle_name: str):
    metrics = parse_dsg_cst_results(**_bundle_paths(bundle_name))
    assert metrics["Kc_TM21_mean"] > 7.0
    assert 0.0 < metrics["vp_std"] < 0.2
    assert metrics["S11_max"] <= -16.0
    assert metrics["S21_mean"] < 0.0
    assert metrics["ohmic_loss_mean"] > 0.0

    dispersion = parse_dsg_dispersion(_bundle_paths(bundle_name)["dispersion_tm21_path"])
    assert dispersion["freq_ghz"].between(96.0, 110.0).all()
    assert "vp_norm" in dispersion.columns


def test_postprocessing_missing_file(tmp_path: Path):
    missing_bundle = {
        "dispersion_tm21_path": tmp_path / "dispersion_tm21.txt",
        "dispersion_fundamental_path": tmp_path / "dispersion_fundamental.txt",
        "kc_tm21_path": tmp_path / "kc_tm21.txt",
        "kc_fundamental_path": tmp_path / "kc_fundamental.txt",
        "sparameters_path": tmp_path / "sparameters.txt",
    }
    with pytest.raises(PostprocessingFileMissingError):
        parse_dsg_cst_results(**missing_bundle)


def test_postprocessing_invalid_columns():
    with pytest.raises(PostprocessingFormatError):
        parse_dsg_cst_results(**_bundle_paths("dsg_missing_columns"))


def test_postprocessing_empty_band():
    with pytest.raises(PostprocessingEmptyBandError):
        parse_dsg_cst_results(**_bundle_paths("dsg_out_of_band"))


def test_postprocessing_empty_file():
    with pytest.raises(PostprocessingFormatError):
        parse_dsg_cst_results(**_bundle_paths("dsg_empty"))


def test_postprocessing_nan_detection():
    with pytest.raises(PostprocessingFormatError):
        parse_dsg_cst_results(**_bundle_paths("dsg_nan"))


def test_postprocessing_unit_conversion():
    dispersion = parse_dsg_dispersion(_bundle_paths("dsg")["dispersion_tm21_path"])
    assert dispersion["freq_ghz"].iloc[1] == pytest.approx(98.0)
