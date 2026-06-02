"""Robust parsers for CST-exported postprocessing files."""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


class PostprocessingError(RuntimeError):
    """Base error for CST postprocessing failures."""


class PostprocessingFileMissingError(PostprocessingError, FileNotFoundError):
    """Raised when an expected export file is missing."""


class PostprocessingFormatError(PostprocessingError):
    """Raised when a postprocessing file is malformed."""


class PostprocessingEmptyBandError(PostprocessingError):
    """Raised when no samples lie inside the target frequency band."""


class PartialResultError(PostprocessingError):
    """Raised when only a subset of required exports is present."""


COMMENT_PREFIXES = ("#", "%", "//")
FREQ_COLUMN = "freq_ghz"
KC_COLUMN = "Kc"
S11_COLUMN = "S11_dB"
S21_COLUMN = "S21_dB"
PHASE_SHIFT_COLUMN = "phase_shift"
VP_NORM_COLUMN = "vp_norm"

CANONICAL_ALIASES: dict[str, set[str]] = {
    FREQ_COLUMN: {
        "f",
        "freq",
        "frequency",
        "freq_ghz",
        "frequency_ghz",
        "ghz",
        "freq_hz",
        "frequency_hz",
        "hz",
        "freq_mhz",
        "frequency_mhz",
        "mhz",
        "freq_khz",
        "frequency_khz",
        "khz",
    },
    KC_COLUMN: {"kc", "k_c", "kc_ohm"},
    S11_COLUMN: {"s11", "s11_db", "s11db"},
    S21_COLUMN: {"s21", "s21_db", "s21db"},
    PHASE_SHIFT_COLUMN: {"phase", "phase_shift", "phi", "phase_deg"},
    VP_NORM_COLUMN: {"vp_norm", "vp_over_c", "vp_c", "phase_velocity_norm", "vpoverb"},
}

FREQ_UNIT_HINTS = {
    "freq_ghz": 1.0,
    "frequency_ghz": 1.0,
    "ghz": 1.0,
    "freq_hz": 1e-9,
    "frequency_hz": 1e-9,
    "hz": 1e-9,
    "freq_mhz": 1e-3,
    "frequency_mhz": 1e-3,
    "mhz": 1e-3,
    "freq_khz": 1e-6,
    "frequency_khz": 1e-6,
    "khz": 1e-6,
}


def _clean_token(token: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", token.strip().lower()).strip("_")


def _is_numeric_row(tokens: Iterable[str]) -> bool:
    try:
        for token in tokens:
            float(token)
        return True
    except ValueError:
        return False


def _infer_delimiter(sample_line: str) -> str:
    return r"\s*,\s*" if "," in sample_line else r"\s+"


def _read_raw_table(path: str | Path, expected_columns: list[str]) -> tuple[pd.DataFrame, bool, str | None]:
    path_obj = Path(path)
    if not path_obj.exists():
        raise PostprocessingFileMissingError(f"Postprocessing file not found: {path_obj}")

    text = path_obj.read_text(encoding="utf-8").strip()
    if not text:
        raise PostprocessingFormatError(f"Postprocessing file is empty: {path_obj}")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    content_lines = [line for line in lines if not line.startswith(COMMENT_PREFIXES)]
    if not content_lines:
        raise PostprocessingFormatError(f"Postprocessing file has no data rows: {path_obj}")

    delimiter = _infer_delimiter(content_lines[0])
    first_tokens = re.split(delimiter, content_lines[0])
    has_header = not _is_numeric_row(first_tokens)

    csv_buffer = io.StringIO("\n".join(content_lines))
    if has_header:
        df = pd.read_csv(csv_buffer, sep=delimiter, engine="python")
        raw_columns = [str(col) for col in df.columns]
    else:
        df = pd.read_csv(csv_buffer, sep=delimiter, engine="python", header=None)
        raw_columns = expected_columns[: df.shape[1]]
        df.columns = raw_columns
    return df, has_header, raw_columns[0] if raw_columns else None


def _canonicalize_columns(df: pd.DataFrame, expected_columns: list[str], path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    rename_map: dict[str, str] = {}
    matched: dict[str, str] = {}
    for column in df.columns:
        cleaned = _clean_token(str(column))
        canonical_name = None
        for expected in expected_columns:
            if cleaned == expected or cleaned in CANONICAL_ALIASES.get(expected, set()):
                canonical_name = expected
                break
        if canonical_name is not None:
            rename_map[column] = canonical_name
            matched[canonical_name] = cleaned

    if rename_map:
        df = df.rename(columns=rename_map)

    missing = [column for column in expected_columns if column not in df.columns]
    if missing:
        raise PostprocessingFormatError(
            f"Missing required columns {missing} in {path}. Available columns: {list(df.columns)}"
        )

    return df.loc[:, expected_columns].copy(), matched


def _infer_frequency_scale(freq_alias: str | None, values: np.ndarray) -> float:
    if freq_alias and freq_alias in FREQ_UNIT_HINTS:
        return FREQ_UNIT_HINTS[freq_alias]

    median_abs = float(np.nanmedian(np.abs(values)))
    if median_abs > 1e8:
        return 1e-9
    if median_abs > 1e5:
        return 1e-6
    if median_abs > 1e3:
        return 1e-3
    return 1.0


def _finalize_numeric_columns(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        if df[column].isna().any():
            raise PostprocessingFormatError(f"NaN or non-numeric values detected in column '{column}' of {path}")
    if not np.isfinite(df.to_numpy(dtype=float)).all():
        raise PostprocessingFormatError(f"Non-finite values detected in {path}")
    return df


def _load_curve(path: str | Path, expected_columns: list[str]) -> pd.DataFrame:
    path_obj = Path(path)
    raw_df, _, first_column_alias = _read_raw_table(path_obj, expected_columns)
    standardized_df, matched_aliases = _canonicalize_columns(raw_df, expected_columns, path_obj)
    standardized_df = _finalize_numeric_columns(standardized_df, path_obj)

    freq_alias = matched_aliases.get(FREQ_COLUMN, _clean_token(first_column_alias) if first_column_alias else None)
    scale = _infer_frequency_scale(freq_alias, standardized_df[FREQ_COLUMN].to_numpy(dtype=float))
    standardized_df[FREQ_COLUMN] = standardized_df[FREQ_COLUMN].to_numpy(dtype=float) * scale
    standardized_df = standardized_df.sort_values(FREQ_COLUMN).reset_index(drop=True)
    return standardized_df


def parse_sparameters(path: str | Path) -> pd.DataFrame:
    """Parse S-parameter export into a standardized dataframe."""

    return _load_curve(path, [FREQ_COLUMN, S11_COLUMN, S21_COLUMN])


def select_frequency_band(
    df: pd.DataFrame,
    band_ghz: tuple[float, float],
    *,
    freq_column: str = FREQ_COLUMN,
    source: str = "curve",
) -> pd.DataFrame:
    """Select the requested frequency band or raise a clear error."""

    lower, upper = band_ghz
    band_df = df[(df[freq_column] >= lower) & (df[freq_column] <= upper)].reset_index(drop=True)
    if band_df.empty:
        raise PostprocessingEmptyBandError(
            f"No {source} samples found inside {lower:.1f}-{upper:.1f} GHz."
        )
    return band_df


def validate_required_post_files(file_map: dict[str, str | Path]) -> dict[str, Path]:
    """Validate that all required exported files exist."""

    resolved = {name: Path(path) for name, path in file_map.items()}
    existing = {name: path for name, path in resolved.items() if path.exists()}
    if not existing:
        missing_list = ", ".join(str(path) for path in resolved.values())
        raise PostprocessingFileMissingError(f"No postprocessing exports found. Expected: {missing_list}")
    if len(existing) != len(resolved):
        missing = [name for name, path in resolved.items() if not path.exists()]
        raise PartialResultError(f"Partial CST exports detected. Missing files for: {missing}")
    return resolved


def parse_dsg_dispersion(path: str | Path) -> pd.DataFrame:
    """Parse DSG dispersion export for one mode."""

    df = _load_curve(path, [FREQ_COLUMN, PHASE_SHIFT_COLUMN, VP_NORM_COLUMN])
    return df


def parse_dsg_coupling_impedance(path: str | Path) -> pd.DataFrame:
    """Parse DSG coupling-impedance export."""

    return _load_curve(path, [FREQ_COLUMN, KC_COLUMN])


def parse_dsg_sparameters(path: str | Path) -> pd.DataFrame:
    """Parse DSG finite-length S-parameters."""

    return parse_sparameters(path)


def _electron_velocity_norm(beam_voltage_kv: float) -> float:
    voltage = float(beam_voltage_kv) * 1e3
    gamma = 1.0 + voltage / 5.11e5
    beta = np.sqrt(max(0.0, 1.0 - 1.0 / (gamma**2)))
    return float(beta)


def compute_sync_error(
    dispersion_df: pd.DataFrame,
    target_frequency_ghz: float = 100.0,
    beam_voltage_kv: float = 5.45,
    search_window_ghz: float = 2.5,
) -> tuple[float, float]:
    """Compute DSG sync error at the target frequency."""

    df = dispersion_df.copy()
    local_df = df[np.abs(df[FREQ_COLUMN] - target_frequency_ghz) <= search_window_ghz]
    if local_df.empty:
        raise PostprocessingEmptyBandError(
            f"No dispersion samples found near target frequency {target_frequency_ghz:.1f} GHz."
        )
    idx = (local_df[FREQ_COLUMN] - target_frequency_ghz).abs().idxmin()
    vp_norm = float(local_df.loc[idx, VP_NORM_COLUMN])
    sync_error = abs(vp_norm - _electron_velocity_norm(beam_voltage_kv))
    return sync_error, vp_norm


def compute_mode_ratio(
    kc_tm21_df: pd.DataFrame,
    kc_fund_df: pd.DataFrame,
    target_frequency_ghz: float = 100.0,
    search_window_ghz: float = 3.0,
) -> float:
    """Compute the TM21-to-fundamental mode ratio near the target frequency."""

    tm21_local = kc_tm21_df[np.abs(kc_tm21_df[FREQ_COLUMN] - target_frequency_ghz) <= search_window_ghz]
    fund_local = kc_fund_df[np.abs(kc_fund_df[FREQ_COLUMN] - target_frequency_ghz) <= search_window_ghz]
    if tm21_local.empty or fund_local.empty:
        raise PostprocessingEmptyBandError(
            f"Unable to compute mode ratio near {target_frequency_ghz:.1f} GHz."
        )
    tm21_value = float(tm21_local.iloc[(tm21_local[FREQ_COLUMN] - target_frequency_ghz).abs().argmin()][KC_COLUMN])
    fund_value = float(fund_local.iloc[(fund_local[FREQ_COLUMN] - target_frequency_ghz).abs().argmin()][KC_COLUMN])
    return tm21_value / max(fund_value, 1e-9)


def parse_dsg_cst_results(
    *,
    dispersion_tm21_path: str | Path,
    dispersion_fundamental_path: str | Path,
    kc_tm21_path: str | Path,
    kc_fundamental_path: str | Path,
    sparameters_path: str | Path,
    working_band_ghz: tuple[float, float] = (96.0, 110.0),
    target_frequency_ghz: float = 100.0,
    beam_voltage_kv: float = 5.45,
) -> dict[str, float]:
    """Parse DSG eigenmode and cold-test results into BO-facing metrics."""

    validated = validate_required_post_files(
        {
            "dispersion_tm21": dispersion_tm21_path,
            "dispersion_fundamental": dispersion_fundamental_path,
            "kc_tm21": kc_tm21_path,
            "kc_fundamental": kc_fundamental_path,
            "sparameters": sparameters_path,
        }
    )
    tm21_dispersion = parse_dsg_dispersion(validated["dispersion_tm21"])
    fundamental_dispersion = parse_dsg_dispersion(validated["dispersion_fundamental"])
    kc_tm21 = parse_dsg_coupling_impedance(validated["kc_tm21"])
    kc_fund = parse_dsg_coupling_impedance(validated["kc_fundamental"])
    sparams = parse_dsg_sparameters(validated["sparameters"])

    band_tm21 = select_frequency_band(tm21_dispersion, working_band_ghz, source="TM21 dispersion")
    band_fund = select_frequency_band(fundamental_dispersion, working_band_ghz, source="fundamental dispersion")
    band_kc_tm21 = select_frequency_band(kc_tm21, working_band_ghz, source="TM21 coupling impedance")
    band_kc_fund = select_frequency_band(kc_fund, working_band_ghz, source="fundamental coupling impedance")
    band_sparams = select_frequency_band(sparams, working_band_ghz, source="DSG S-parameters")

    sync_error, vp_norm = compute_sync_error(
        band_tm21,
        target_frequency_ghz=target_frequency_ghz,
        beam_voltage_kv=beam_voltage_kv,
    )
    mode_ratio = compute_mode_ratio(
        band_kc_tm21,
        band_kc_fund,
        target_frequency_ghz=target_frequency_ghz,
    )
    tm21_idx = (band_tm21[FREQ_COLUMN] - target_frequency_ghz).abs().idxmin()
    fund_idx = (band_fund[FREQ_COLUMN] - target_frequency_ghz).abs().idxmin()
    s21_mean = float(band_sparams[S21_COLUMN].mean())

    return {
        "Kc_TM21_mean": float(band_kc_tm21[KC_COLUMN].mean()),
        "Kc_mean": float(band_kc_tm21[KC_COLUMN].mean()),
        "sync_error": float(sync_error),
        "vp_TM21_std": float(band_tm21[VP_NORM_COLUMN].std(ddof=0)),
        "vp_std": float(sync_error),
        "ohmic_loss_mean": float(max(0.0, -s21_mean)),
        "S11_max": float(band_sparams[S11_COLUMN].max()),
        "S21_mean": s21_mean,
        "mode_ratio": float(mode_ratio),
        "f_TM21_ghz": float(band_tm21.loc[tm21_idx, FREQ_COLUMN]),
        "f_fund_ghz": float(band_fund.loc[fund_idx, FREQ_COLUMN]),
        "vp_norm_target": float(vp_norm),
    }
