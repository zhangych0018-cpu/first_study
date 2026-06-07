"""本模块负责把真实 CST 结果树中的 DSG 冷结构结果导出为项目统一的后处理文件。

输入是 `cst.results.ProjectFile(...).get_3d()` 返回的结果访问对象、结果树节点列表、导出目录以及可选的节点映射；
输出是 `dispersion_tm21.txt`、`dispersion_fundamental.txt`、`kc_tm21.txt`、`kc_fundamental.txt`、
`sparameters.txt` 和可选的 `mode_frequencies.csv`。它位于真实 CST 自动化和 BO 后端之间，
让单点 CST 检查脚本与贝叶斯优化主循环共享同一套导出规则，避免两条链路的文件格式漂移。
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Iterable


STANDARD_EXPORT_KEYS = (
    "dispersion_tm21",
    "dispersion_fundamental",
    "kc_tm21",
    "kc_fundamental",
    "sparameters",
)

OPTIONAL_EXPORT_KEYS = ("mode_frequencies",)

DEFAULT_FILENAMES = {
    "dispersion_tm21": "dispersion_tm21.txt",
    "dispersion_fundamental": "dispersion_fundamental.txt",
    "kc_tm21": "kc_tm21.txt",
    "kc_fundamental": "kc_fundamental.txt",
    "sparameters": "sparameters.txt",
    "mode_frequencies": "mode_frequencies.csv",
}

STANDARD_COLUMNS = {
    "dispersion_tm21": ["freq_ghz", "phase_shift", "vp_norm"],
    "dispersion_fundamental": ["freq_ghz", "phase_shift", "vp_norm"],
    "kc_tm21": ["freq_ghz", "Kc"],
    "kc_fundamental": ["freq_ghz", "Kc"],
}

AUTO_DETECT_KEYWORDS = {
    "dispersion_tm21": [
        ("dispersion", "tm21"),
        ("phase shift", "tm21"),
        ("phase velocity", "tm21"),
        ("vp", "tm21"),
    ],
    "dispersion_fundamental": [
        ("dispersion", "fundamental"),
        ("phase shift", "fundamental"),
        ("phase velocity", "fundamental"),
        ("vp", "fundamental"),
        ("dispersion", "mode 1"),
    ],
    "kc_tm21": [
        ("coupling", "impedance", "tm21"),
        ("interaction", "impedance", "tm21"),
        ("kc", "tm21"),
    ],
    "kc_fundamental": [
        ("coupling", "impedance", "fundamental"),
        ("interaction", "impedance", "fundamental"),
        ("kc", "fundamental"),
        ("coupling", "impedance", "mode 1"),
    ],
    "mode_frequencies": [
        ("mode frequencies",),
        ("eigenmode", "frequency"),
        ("mode", "frequency"),
    ],
}


class CSTResultExportError(RuntimeError):
    """当 CST 结果树无法导出为 BO 所需标准文件时抛出，调用方据此记录明确失败原因。"""


def db20(value: complex | float) -> float:
    magnitude = abs(value)
    return 20.0 * math.log10(max(magnitude, 1e-30))


def flatten_sample(sample: Any) -> list[Any]:
    if isinstance(sample, (tuple, list)):
        return list(sample)
    return [sample]


def scalar_value(value: Any) -> float:
    """把 CST 结果项中的实数或复数压成可写入标准曲线的标量，复数默认取幅值。"""

    if isinstance(value, complex):
        return float(abs(value))
    return float(value)


def _write_rows(path: Path, headers: list[str], rows: Iterable[Iterable[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def combine_sparameter_rows(
    s11_data: list[tuple[Any, ...]],
    s21_data: list[tuple[Any, ...]],
) -> list[dict[str, float]]:
    """合并 CST 的 `S1,1` 和 `S2,1` 节点，得到 BO parser 需要的频率、S11 和 S21 dB 曲线。"""

    rows: list[dict[str, float]] = []
    count = min(len(s11_data), len(s21_data))
    for idx in range(count):
        f1, s11_value, *s11_rest = s11_data[idx]
        f2, s21_value, *s21_rest = s21_data[idx]
        if abs(float(f1) - float(f2)) > 1e-6:
            raise CSTResultExportError(f"S-parameter frequency grids do not align at index {idx}: {f1} vs {f2}")
        row = {
            "freq_ghz": float(f1),
            "S11_real": float(complex(s11_value).real),
            "S11_imag": float(complex(s11_value).imag),
            "S11_abs": float(abs(complex(s11_value))),
            "S11_dB": float(db20(complex(s11_value))),
            "S21_real": float(complex(s21_value).real),
            "S21_imag": float(complex(s21_value).imag),
            "S21_abs": float(abs(complex(s21_value))),
            "S21_dB": float(db20(complex(s21_value))),
        }
        if s11_rest:
            row["S11_aux_real"] = float(complex(s11_rest[0]).real)
            row["S11_aux_imag"] = float(complex(s11_rest[0]).imag)
        if s21_rest:
            row["S21_aux_real"] = float(complex(s21_rest[0]).real)
            row["S21_aux_imag"] = float(complex(s21_rest[0]).imag)
        rows.append(row)
    if not rows:
        raise CSTResultExportError("S-parameter result items are empty.")
    return rows


def export_sparameters(results_3d: Any, run_dir: Path, filename: str = "sparameters.txt") -> dict[str, Any]:
    """从固定端口结果节点导出标准 S 参数文件，并额外保留复数 CSV 方便排障。"""

    s11_item = r"1D Results\S-Parameters\S1,1"
    s21_item = r"1D Results\S-Parameters\S2,1"
    s11_data = results_3d.get_result_item(s11_item).get_data()
    s21_data = results_3d.get_result_item(s21_item).get_data()
    rows = combine_sparameter_rows(s11_data, s21_data)

    txt_path = run_dir / filename
    _write_rows(txt_path, ["freq_ghz", "S11_dB", "S21_dB"], ([row["freq_ghz"], row["S11_dB"], row["S21_dB"]] for row in rows))

    csv_path = run_dir / "sparameters_complex.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return {
        "s11_item": s11_item,
        "s21_item": s21_item,
        "sparameters_txt": str(txt_path),
        "sparameters_complex_csv": str(csv_path),
        "sample_count": len(rows),
    }


def export_standard_curve(results_3d: Any, tree_item: str, output_path: Path, columns: list[str]) -> dict[str, Any]:
    """按给定列名把一个 CST 1D 节点写成 parser 可直接读取的文本曲线。

    真实 CST 项目里的派生结果通常已经把频率和目标物理量打包在同一条 1D 曲线里；这里只做轻量标准化：
    取每行前 N 个值并写入约定列名。若节点列数不够，说明模板导出的不是完整 BO 指标，直接失败而不是猜测补值。
    """

    data = results_3d.get_result_item(tree_item).get_data()
    if not isinstance(data, list) or not data:
        raise CSTResultExportError(f"Result item is empty or not a 1D data list: {tree_item}")

    rows = []
    for sample in data:
        flat = flatten_sample(sample)
        if len(flat) < len(columns):
            raise CSTResultExportError(
                f"Result item {tree_item} has {len(flat)} columns, but {len(columns)} are required for {output_path.name}."
            )
        rows.append([scalar_value(value) for value in flat[: len(columns)]])
    _write_rows(output_path, columns, rows)
    return {"tree_item": tree_item, "path": str(output_path), "sample_count": len(rows), "columns": columns}


def find_matching_tree_item(tree_items: list[str], keyword_groups: list[tuple[str, ...]]) -> str | None:
    """按一组关键词候选从结果树里挑选第一个匹配节点，用于没有显式配置映射时的保守自动发现。"""

    lowered = [(item.lower(), item) for item in tree_items]
    for keywords in keyword_groups:
        for lowered_item, original in lowered:
            if all(keyword.lower() in lowered_item for keyword in keywords):
                return original
    return None


def resolve_result_tree_items(
    tree_items: list[str],
    configured_items: dict[str, str] | None = None,
) -> dict[str, str | None]:
    """合并用户显式配置和关键词自动发现，得到各标准导出文件对应的 CST 结果树节点。"""

    configured_items = configured_items or {}
    resolved: dict[str, str | None] = {}
    for key in (*STANDARD_EXPORT_KEYS, *OPTIONAL_EXPORT_KEYS):
        if key == "sparameters":
            resolved[key] = r"1D Results\S-Parameters\S1,1" if r"1D Results\S-Parameters\S1,1" in tree_items else None
            continue
        explicit = configured_items.get(key)
        if explicit:
            resolved[key] = explicit if explicit in tree_items else None
        else:
            resolved[key] = find_matching_tree_item(tree_items, AUTO_DETECT_KEYWORDS.get(key, []))
    return resolved


def export_standard_dsg_results(
    *,
    results_3d: Any,
    tree_items: list[str],
    run_dir: Path,
    filenames: dict[str, str] | None = None,
    result_tree_items: dict[str, str] | None = None,
) -> dict[str, Any]:
    """把真实 CST 结果树导出成 BO 标准文件，并返回完整映射与缺失信息。"""

    filenames = {**DEFAULT_FILENAMES, **(filenames or {})}
    resolved_items = resolve_result_tree_items(tree_items, result_tree_items)
    exported: dict[str, Any] = {}
    missing: dict[str, str] = {}

    try:
        exported["sparameters"] = export_sparameters(results_3d, run_dir, filenames["sparameters"])
    except Exception as exc:
        missing["sparameters"] = str(exc)

    for key in STANDARD_EXPORT_KEYS:
        if key == "sparameters":
            continue
        tree_item = resolved_items.get(key)
        if not tree_item:
            missing[key] = "没有在 CST 结果树中找到匹配节点；请在配置中设置 cst.postprocessing.result_tree_items。"
            continue
        try:
            exported[key] = export_standard_curve(results_3d, tree_item, run_dir / filenames[key], STANDARD_COLUMNS[key])
        except Exception as exc:
            missing[key] = str(exc)

    mode_item = resolved_items.get("mode_frequencies")
    if mode_item:
        try:
            exported["mode_frequencies"] = export_standard_curve(
                results_3d,
                mode_item,
                run_dir / filenames["mode_frequencies"],
                ["mode_index", "freq_ghz"],
            )
        except Exception as exc:
            missing["mode_frequencies"] = str(exc)

    return {
        "exported": exported,
        "missing": missing,
        "resolved_tree_items": resolved_items,
        "required_complete": all(key in exported for key in STANDARD_EXPORT_KEYS),
    }
