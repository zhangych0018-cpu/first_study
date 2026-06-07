"""本模块负责解析 CST 导出的文本结果，包括 DSG 所需的色散、耦合阻抗与 S 参数文件，并把它们整理成优化器可直接消费的指标。"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


class PostprocessingError(RuntimeError):
    """所有 CST 后处理异常的基类，用于统一捕获和区分导出解析阶段的问题。"""


class PostprocessingFileMissingError(PostprocessingError, FileNotFoundError):
    """在期望的导出文件不存在时抛出，帮助清楚区分没有导出来和格式不对两类问题。"""


class PostprocessingFormatError(PostprocessingError):
    """在后处理文件格式错误、列缺失或含有非数值内容时抛出，阻止错误数据继续流入优化流程。"""


class PostprocessingEmptyBandError(PostprocessingError):
    """当目标频带内没有有效采样点时抛出，用于提醒结果导出范围与工程要求不一致。"""


class PartialResultError(PostprocessingError):
    """当只导出了一部分必需文件时抛出，避免后续在信息不完整的情况下继续计算指标。"""


COMMENT_PREFIXES = ("#", "%", "//")
FREQ_COLUMN = "freq_ghz"
KC_COLUMN = "Kc"
S11_COLUMN = "S11_dB"
S21_COLUMN = "S21_dB"
PHASE_SHIFT_COLUMN = "phase_shift"
VP_NORM_COLUMN = "vp_norm"
MODE_INDEX_COLUMN = "mode_index"

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
    """把 S 参数导出文件解析为标准 DataFrame，并统一频率列与 S 参数列命名。"""

    return _load_curve(path, [FREQ_COLUMN, S11_COLUMN, S21_COLUMN])


def select_frequency_band(
    df: pd.DataFrame,
    band_ghz: tuple[float, float],
    *,
    freq_column: str = FREQ_COLUMN,
    source: str = "curve",
) -> pd.DataFrame:
    """从完整曲线中截取目标频带范围，如果该频带内没有数据就给出明确异常。"""

    lower, upper = band_ghz
    band_df = df[(df[freq_column] >= lower) & (df[freq_column] <= upper)].reset_index(drop=True)
    if band_df.empty:
        raise PostprocessingEmptyBandError(
            f"No {source} samples found inside {lower:.1f}-{upper:.1f} GHz."
        )
    return band_df


def validate_required_post_files(file_map: dict[str, str | Path]) -> dict[str, Path]:
    """检查一组必须的后处理文件是否齐全，并在缺失或只存在部分文件时返回清晰错误。"""

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
    """解析 DSG 色散曲线文件，统一得到频率、相移和归一化相速度三列。"""

    df = _load_curve(path, [FREQ_COLUMN, PHASE_SHIFT_COLUMN, VP_NORM_COLUMN])
    return df


def parse_dsg_coupling_impedance(path: str | Path) -> pd.DataFrame:
    """解析 DSG 耦合阻抗曲线文件，并输出标准化后的频率与 `Kc` 数据表。"""

    return _load_curve(path, [FREQ_COLUMN, KC_COLUMN])


def parse_dsg_sparameters(path: str | Path) -> pd.DataFrame:
    """解析 DSG 有限长结构的 S 参数文件，为后续指标计算提供统一输入格式。"""

    return parse_sparameters(path)


def parse_mode_frequencies(path: str | Path) -> pd.DataFrame:
    """解析可选的 CST mode-frequency 导出文件，用于在 BO 记录中保留模式识别和本征频率诊断信息。"""

    path_obj = Path(path)
    raw_df, has_header, _ = _read_raw_table(path_obj, [MODE_INDEX_COLUMN, FREQ_COLUMN])
    if has_header:
        rename_map = {}
        for column in raw_df.columns:
            cleaned = _clean_token(str(column))
            if cleaned in {"mode", "mode_index", "mode_number", "index"}:
                rename_map[column] = MODE_INDEX_COLUMN
            elif cleaned in CANONICAL_ALIASES[FREQ_COLUMN]:
                rename_map[column] = FREQ_COLUMN
        raw_df = raw_df.rename(columns=rename_map)
    else:
        raw_df.columns = [MODE_INDEX_COLUMN, FREQ_COLUMN][: raw_df.shape[1]]

    missing = [column for column in [MODE_INDEX_COLUMN, FREQ_COLUMN] if column not in raw_df.columns]
    if missing:
        raise PostprocessingFormatError(
            f"Missing required mode-frequency columns {missing} in {path_obj}. Available columns: {list(raw_df.columns)}"
        )
    df = _finalize_numeric_columns(raw_df[[MODE_INDEX_COLUMN, FREQ_COLUMN]].copy(), path_obj)
    scale = _infer_frequency_scale(None, df[FREQ_COLUMN].to_numpy(dtype=float))
    df[FREQ_COLUMN] = df[FREQ_COLUMN].to_numpy(dtype=float) * scale
    return df.sort_values(MODE_INDEX_COLUMN).reset_index(drop=True)


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
    """在目标频率附近计算同步误差，并返回对应的归一化相速度，供 DSG 目标函数使用。"""

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
    """在目标频率附近计算 TM21 模与基模的耦合阻抗比值，用于模式竞争约束分析。"""

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
    mode_frequencies_path: str | Path | None = None,
    working_band_ghz: tuple[float, float] = (96.0, 110.0),
    target_frequency_ghz: float = 100.0,
    beam_voltage_kv: float = 5.45,
) -> dict[str, float]:
    """把 DSG 的色散、耦合阻抗和 S 参数导出文件整合为优化器直接使用的 BO 指标字典。"""

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

    metrics = {
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
    if mode_frequencies_path is not None and Path(mode_frequencies_path).exists():
        mode_frequencies = parse_mode_frequencies(mode_frequencies_path)
        metrics.update(
            {
                "mode_frequency_count": float(len(mode_frequencies)),
                "mode_frequency_min_ghz": float(mode_frequencies[FREQ_COLUMN].min()),
                "mode_frequency_max_ghz": float(mode_frequencies[FREQ_COLUMN].max()),
                "mode1_frequency_ghz": float(mode_frequencies[FREQ_COLUMN].iloc[0]),
            }
        )
    return metrics


def parse_dsg_sparameter_results(
    *,
    sparameters_path: str | Path,
    working_band_ghz: tuple[float, float] = (96.0, 110.0),
) -> dict[str, Any]:
    """只基于当前真实 CST 已能稳定导出的 S 参数计算 BO 指标。

    这个降参路径用于真实工程暂时缺少 dispersion/Kc/mode-frequency 标准节点时的完整流程联调。
    五个 DSG 几何变量仍由 BO 修改；减少的是后处理依赖的物理输出数量：
    用 `S21_mean` 代表传输质量，用 `S11` 起伏代表匹配稳定性，用插入损耗和 `S11_max` 完成目标与约束。
    """

    validated = validate_required_post_files({"sparameters": sparameters_path})
    sparams = parse_dsg_sparameters(validated["sparameters"])
    band_sparams = select_frequency_band(sparams, working_band_ghz, source="DSG S-parameters")

    s11_max = float(band_sparams[S11_COLUMN].max())
    s11_min = float(band_sparams[S11_COLUMN].min())
    s11_mean = float(band_sparams[S11_COLUMN].mean())
    s11_ripple = float(band_sparams[S11_COLUMN].std(ddof=0))
    s21_mean = float(band_sparams[S21_COLUMN].mean())
    s21_min = float(band_sparams[S21_COLUMN].min())
    insertion_loss_mean = float(max(0.0, -s21_mean))

    return {
        "Kc_mean": s21_mean,
        "Kc_TM21_mean": s21_mean,
        "vp_std": s11_ripple,
        "vp_TM21_std": s11_ripple,
        "sync_error": s11_ripple,
        "ohmic_loss_mean": insertion_loss_mean,
        "S11_max": s11_max,
        "S21_mean": s21_mean,
        "S21_min": s21_min,
        "S11_min": s11_min,
        "S11_mean": s11_mean,
        "S11_ripple": s11_ripple,
        "insertion_loss_mean": insertion_loss_mean,
        "postprocessing_mode": "sparameter_only",
    }
