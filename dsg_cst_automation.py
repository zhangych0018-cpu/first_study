"""本模块负责把真实 DSG-SWS 的 CST 工程包装成一个可批量调用的自动化脚本。

当前版本不修改几何建模逻辑，只做以下几类工程化能力：

1. 识别当前 CST 工程中已经存在的参数名与参数文件；
2. 通过 Python 代码修改工程参数，而不是手工在 CST 工作区逐个输入；
3. 支持单次参数覆盖、规则扫描和外部 case 表驱动的循环仿真；
4. 对每个 case 运行 solver、导出结果树、输出标准化结果文件；
5. 记录批量运行摘要，并在需要时把工程参数恢复到初始基线。

它是“真实 CST 工程”和“后续贝叶斯优化/批量扫描”之间的桥接脚本。
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sws_bo.utils.cst_result_export import export_standard_dsg_results


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_PATH = Path(r"C:\Users\87007\Desktop\SWS\DSG_SWS.cst")
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "dsg_cst_exports"


@dataclass
class ProjectParameter:
    """描述一个从 CST 工程识别到的参数。"""

    name: str
    expr: str | None = None
    value: str | None = None
    descr: str | None = None
    source: str = "unknown"


@dataclass
class ParameterCase:
    """描述一次要送入 CST 的参数方案。"""

    case_name: str
    parameter_values: dict[str, str]
    changed_parameters: dict[str, str]


def ensure_cst_python_paths(cli_override: Path | None) -> Path:
    """把本机 CST Python 库目录注入 `sys.path`，并返回可用的 AMD64 安装目录。"""

    candidates = [
        cli_override,
        Path(os.environ["CST_AMD64_DIR"]) if os.environ.get("CST_AMD64_DIR") else None,
        Path(r"D:\Programs\CST\AMD64"),
        Path(r"C:\Program Files\CST Studio Suite 2024\AMD64"),
        Path(r"C:\Program Files\CST Studio Suite 2025\AMD64"),
    ]

    for amd64_dir in candidates:
        if amd64_dir is None:
            continue
        lib_dir = amd64_dir / "python_cst_libraries"
        if amd64_dir.exists() and lib_dir.exists():
            for entry in (str(lib_dir), str(amd64_dir)):
                if entry not in sys.path:
                    sys.path.insert(0, entry)
            return amd64_dir

    raise FileNotFoundError(
        "无法定位 CST Python 库目录，请通过 --cst-amd64 或环境变量 CST_AMD64_DIR 指定。"
    )


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "对真实 DSG-SWS 的 CST 工程执行自动化流程：识别参数、修改参数、批量循环求解并导出结果。"
        )
    )
    parser.add_argument("--project-path", type=Path, default=DEFAULT_PROJECT_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cst-amd64", type=Path, default=None)
    parser.add_argument(
        "--skip-solver",
        action="store_true",
        help="不重新启动求解器，直接读取当前工程已有结果并导出。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只识别参数和计划运行的 case，不写回参数，也不真正启动 solver。",
    )
    parser.add_argument(
        "--list-parameters",
        action="store_true",
        help="打印当前工程识别到的参数清单。可与 --dry-run 联合使用。",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=5.0,
        help="当 solver 异步返回时，轮询运行状态的时间间隔（秒）。",
    )
    parser.add_argument(
        "--result-tree-map",
        type=Path,
        default=None,
        help="可选 JSON 文件，用于显式指定 dispersion/Kc/mode-frequency 对应的结果树节点。",
    )
    parser.add_argument(
        "--set-param",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="给所有 case 统一施加的参数覆盖，可重复传入。",
    )
    parser.add_argument(
        "--sweep",
        action="append",
        default=[],
        metavar="NAME:START:STOP:COUNT",
        help="规则扫描定义，可重复传入；多个扫描会做笛卡尔积。",
    )
    parser.add_argument(
        "--case-file",
        type=Path,
        default=None,
        help="显式 case 表，支持 JSON 或 CSV。若提供，则优先使用其中的参数覆盖。",
    )
    parser.add_argument(
        "--keep-last-parameters",
        action="store_true",
        help="批量循环结束后保留最后一个 case 的参数，不恢复到初始基线。",
    )
    return parser.parse_args()


def load_cst_modules(cst_amd64_override: Path | None) -> tuple[Any, Any]:
    """导入 `cst.interface` 与 `cst.results`。"""

    amd64_dir = ensure_cst_python_paths(cst_amd64_override)
    print(f"[info] CST AMD64 目录: {amd64_dir}")

    import cst.interface  # type: ignore
    import cst.results  # type: ignore

    return cst.interface, cst.results


def make_run_dir(output_root: Path) -> Path:
    """为本次批量运行创建独立结果目录。"""

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_text(path: Path, text: str) -> None:
    """写出 UTF-8 文本文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    """写出 UTF-8 JSON 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_parameter_json(project_path: Path) -> dict[str, Any]:
    """读取 `.cst` 同名展开目录下的 `Model/Parameters.json`。"""

    parameter_path = project_path.with_suffix("") / "Model" / "Parameters.json"
    if not parameter_path.exists():
        return {"parameter_file_missing": str(parameter_path)}
    return json.loads(parameter_path.read_text(encoding="utf-8"))


def parse_parameter_payload(parameter_payload: dict[str, Any]) -> dict[str, ProjectParameter]:
    """把 `Parameters.json` 解析为按参数名索引的结构化字典。"""

    items = parameter_payload.get("parameters")
    if not isinstance(items, list):
        return {}

    parsed: dict[str, ProjectParameter] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        parsed[name] = ProjectParameter(
            name=name,
            expr=str(item["expr"]) if item.get("expr") is not None else None,
            value=str(item["value"]) if item.get("value") is not None else None,
            descr=str(item["descr"]) if item.get("descr") is not None else None,
            source="parameters.json",
        )
    return parsed


def parameter_to_text(value: Any) -> str:
    """把 Python 值转换为适合写入 CST 参数的字符串。"""

    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def parse_name_value_pair(text: str) -> tuple[str, str]:
    """解析 `NAME=VALUE` 形式的命令行参数。"""

    if "=" not in text:
        raise ValueError(f"参数覆盖格式错误，期望 NAME=VALUE，实际得到: {text}")
    name, raw_value = text.split("=", 1)
    name = name.strip()
    raw_value = raw_value.strip()
    if not name:
        raise ValueError(f"参数名为空: {text}")
    if raw_value == "":
        raise ValueError(f"参数值为空: {text}")
    return name, raw_value


def parse_fixed_overrides(entries: list[str]) -> dict[str, str]:
    """把多个 `--set-param` 转换为字典。"""

    overrides: dict[str, str] = {}
    for entry in entries:
        name, value = parse_name_value_pair(entry)
        overrides[name] = value
    return overrides


def parse_float_text(text: str) -> float:
    """把字符串解析成浮点数。"""

    return float(text.strip())


def build_linspace(start: float, stop: float, count: int) -> list[float]:
    """构造等间距扫描值。"""

    if count <= 0:
        raise ValueError("扫描点数必须大于 0。")
    if count == 1:
        return [start]
    step = (stop - start) / float(count - 1)
    return [start + idx * step for idx in range(count)]


def parse_sweep_specs(entries: list[str]) -> dict[str, list[str]]:
    """解析多个 `NAME:START:STOP:COUNT` 扫描定义。"""

    sweep_map: dict[str, list[str]] = {}
    for entry in entries:
        parts = [item.strip() for item in entry.split(":")]
        if len(parts) != 4:
            raise ValueError(f"扫描格式错误，期望 NAME:START:STOP:COUNT，实际得到: {entry}")
        name, start_text, stop_text, count_text = parts
        values = build_linspace(parse_float_text(start_text), parse_float_text(stop_text), int(count_text))
        sweep_map[name] = [parameter_to_text(value) for value in values]
    return sweep_map


def load_case_file(case_file: Path | None) -> list[dict[str, Any]]:
    """读取显式 case 表，支持 JSON 或 CSV。"""

    if case_file is None:
        return []

    path = case_file.resolve()
    if not path.exists():
        raise FileNotFoundError(f"未找到 case 文件: {path}")

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("JSON case 文件必须是对象列表。")
        return [dict(item) for item in payload if isinstance(item, dict)]

    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            return [dict(row) for row in reader]

    raise ValueError(f"不支持的 case 文件格式: {path.suffix}")


def build_case_definitions(
    *,
    baseline_parameters: dict[str, ProjectParameter],
    fixed_overrides: dict[str, str],
    sweep_map: dict[str, list[str]],
    explicit_case_rows: list[dict[str, Any]],
) -> list[ParameterCase]:
    """根据基线参数、固定覆盖、扫描定义和显式 case 表生成运行计划。"""

    baseline_values = {
        name: parameter_to_text(parameter.value if parameter.value is not None else parameter.expr or "")
        for name, parameter in baseline_parameters.items()
    }

    cases: list[ParameterCase] = []

    def make_case(case_name: str, overrides: dict[str, Any]) -> ParameterCase:
        normalized_overrides = {name: parameter_to_text(value) for name, value in overrides.items() if value not in ("", None)}
        merged = dict(baseline_values)
        merged.update(fixed_overrides)
        merged.update(normalized_overrides)
        changed = {name: value for name, value in merged.items() if baseline_values.get(name) != value}
        return ParameterCase(case_name=case_name, parameter_values=merged, changed_parameters=changed)

    if explicit_case_rows:
        for index, row in enumerate(explicit_case_rows, start=1):
            case_name = str(row.get("case_name") or row.get("name") or f"case_{index:03d}")
            overrides = {
                key: value
                for key, value in row.items()
                if key not in {"case_name", "name"} and value not in ("", None)
            }
            cases.append(make_case(case_name, overrides))
        return cases

    if sweep_map:
        sweep_names = list(sweep_map.keys())
        sweep_value_lists = [sweep_map[name] for name in sweep_names]
        for index, combo in enumerate(itertools.product(*sweep_value_lists), start=1):
            overrides = {name: value for name, value in zip(sweep_names, combo, strict=True)}
            cases.append(make_case(f"case_{index:03d}", overrides))
        return cases

    cases.append(make_case("case_001", {}))
    return cases


def safe_project_save(project: Any, project_path: Path) -> None:
    """兼容不同 CST Python 包签名的保存操作。"""

    save_attempts = (
        lambda: project.save(project_path, allow_overwrite=True),
        lambda: project.save(str(project_path), allow_overwrite=True),
        lambda: project.save(),
    )
    for attempt in save_attempts:
        try:
            attempt()
            return
        except Exception:
            continue


def open_project(cst_interface: Any, project_path: Path) -> tuple[Any, Any]:
    """连接 CST 并打开指定工程。"""

    design_environment = cst_interface.DesignEnvironment.connect_to_any_or_new()
    design_environment.set_quiet_mode(True)
    project = design_environment.open_project(str(project_path))
    return design_environment, project


def close_quietly(project: Any | None, design_environment: Any | None) -> None:
    """尽量安静地关闭项目和 Design Environment。"""

    if project is not None:
        try:
            project.close()
        except Exception:
            pass
    if design_environment is not None:
        try:
            design_environment.close()
        except Exception:
            pass


def list_project_parameter_names(project: Any) -> list[str]:
    """从已打开的 CST 工程里读取参数名列表。"""

    model3d = project.model3d
    count = int(model3d.GetNumberOfParameters())
    names: list[str] = []
    for index in range(count):
        names.append(str(model3d.GetParameterName(index)))
    return sorted(set(names))


def discover_project_parameters(cst_interface: Any, project_path: Path) -> dict[str, ProjectParameter]:
    """综合实时 CST 参数名与 `Parameters.json` 内容，构造当前工程参数目录。"""

    parameter_payload = read_parameter_json(project_path)
    parameters = parse_parameter_payload(parameter_payload)

    design_environment = None
    project = None
    try:
        design_environment, project = open_project(cst_interface, project_path)
        live_names = list_project_parameter_names(project)
        for name in live_names:
            existing = parameters.get(name)
            if existing is None:
                parameters[name] = ProjectParameter(name=name, source="live_cst")
            else:
                existing.source = "live_cst+parameters.json"
    finally:
        close_quietly(project, design_environment)

    return dict(sorted(parameters.items(), key=lambda item: item[0].lower()))


def validate_case_parameters(cases: list[ParameterCase], known_parameters: dict[str, ProjectParameter]) -> None:
    """检查所有 case 的参数名是否都存在于当前工程。"""

    known_names = set(known_parameters.keys())
    unknown_names: set[str] = set()
    for case in cases:
        unknown_names.update(name for name in case.parameter_values if name not in known_names)

    if unknown_names:
        raise ValueError(f"发现工程中不存在的参数名: {sorted(unknown_names)}")


def apply_parameters_to_project(project: Any, parameter_values: dict[str, str]) -> None:
    """通过 CST 的参数接口把一组参数写回当前工程。"""

    model3d = project.model3d
    for name, value in parameter_values.items():
        # 这里优先使用 CST 直接提供的参数接口，而不是改建模宏。
        # 这样可以最大限度保证“只改参数，不改建模历史”。
        model3d.StoreParameter(name, value)

    # 参数写回后主动触发一次重建，使后续 solver 使用新的参数状态。
    try:
        model3d.RebuildOnParametricChange(True, True)
    except Exception:
        pass
    try:
        model3d.Rebuild()
    except Exception:
        try:
            model3d.full_history_rebuild()
        except Exception:
            pass


def run_solver(project: Any, poll_seconds: float) -> dict[str, Any]:
    """启动当前工程求解器，并轮询到 solver 结束。"""

    model3d = project.model3d
    try:
        solver_name = model3d.get_active_solver_name()
    except Exception:
        try:
            solver_name = model3d.GetSolverType()
        except Exception:
            solver_name = "unknown"

    start = time.perf_counter()
    model3d.run_solver()

    while True:
        try:
            running = bool(model3d.is_solver_running())
        except Exception:
            running = False
        if not running:
            break
        time.sleep(max(0.1, poll_seconds))

    return {
        "solver_name": solver_name,
        "run_seconds": time.perf_counter() - start,
    }


def get_tree_items(result_project: Any) -> list[str]:
    """读取并排序整个结果树节点列表。"""

    results_3d = result_project.get_3d()
    items = list(results_3d.get_tree_items())
    items.sort()
    return items


def sanitize_tree_item(tree_item: str) -> str:
    """把结果树节点名称转成适合文件名的文本。"""

    text = tree_item.lower()
    text = text.replace("''", "__doubleprime__")
    text = text.replace("'", "__prime__")
    for old, new in (
        ("\\", "__"),
        ("/", "__"),
        (" ", "_"),
        (",", "_"),
        ("(", "_"),
        (")", "_"),
        ("[", "_"),
        ("]", "_"),
        ("-", "_"),
    ):
        text = text.replace(old, new)
    while "___" in text:
        text = text.replace("___", "__")
    return text.strip("_")


def db20(value: complex | float) -> float:
    """把复数或实数幅值换算成 dB20。"""

    magnitude = abs(value)
    return 20.0 * math.log10(max(magnitude, 1e-30))


def flatten_sample(sample: Any) -> list[Any]:
    """把 CST 单个采样点统一展开为列表。"""

    if isinstance(sample, (tuple, list)):
        return list(sample)
    return [sample]


def build_headers(sample: list[Any]) -> list[str]:
    """根据一个采样点的数据形状构造 CSV 列名。"""

    headers: list[str] = []
    for index, value in enumerate(sample):
        base = "x" if index == 0 else f"value_{index}"
        if isinstance(value, complex):
            headers.extend(
                [
                    f"{base}_real",
                    f"{base}_imag",
                    f"{base}_abs",
                    f"{base}_db20",
                    f"{base}_phase_deg",
                ]
            )
        else:
            headers.append(base)
    return headers


def flatten_value(value: Any) -> list[Any]:
    """把单个标量或复数值展开为 CSV 可写字段。"""

    if isinstance(value, complex):
        return [
            value.real,
            value.imag,
            abs(value),
            db20(value),
            math.degrees(math.atan2(value.imag, value.real)),
        ]
    return [value]


def export_generic_tree_item(results_3d: Any, tree_item: str, output_path: Path) -> bool:
    """把任意可读 1D 节点导出为通用 CSV。"""

    try:
        data = results_3d.get_result_item(tree_item).get_data()
    except Exception:
        return False

    if not isinstance(data, list) or not data:
        return False

    first = flatten_sample(data[0])
    headers = build_headers(first)
    rows = []
    for sample in data:
        flat = flatten_sample(sample)
        row: list[Any] = []
        for value in flat:
            row.extend(flatten_value(value))
        rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)
    return True


def export_all_1d_results(results_3d: Any, tree_items: list[str], run_dir: Path) -> dict[str, str]:
    """批量导出结果树中的全部 1D Results。"""

    exported: dict[str, str] = {}
    used_names: set[str] = set()
    out_dir = run_dir / "all_1d_results"
    for item in tree_items:
        if not item.startswith("1D Results\\"):
            continue
        base_name = sanitize_tree_item(item)
        candidate_name = base_name
        suffix = 2
        while candidate_name in used_names:
            candidate_name = f"{base_name}__{suffix}"
            suffix += 1
        used_names.add(candidate_name)
        output_path = out_dir / f"{candidate_name}.csv"
        if export_generic_tree_item(results_3d, item, output_path):
            exported[item] = str(output_path)
    return exported


def summarize_required_dsg_items(tree_items: list[str]) -> dict[str, Any]:
    """汇总结果树中与 DSG BO 最相关的结果家族。"""

    def find_items(*keywords: str) -> list[str]:
        matches = []
        for item in tree_items:
            lowered_item = item.lower()
            if all(keyword.lower() in lowered_item for keyword in keywords):
                matches.append(item)
        return matches

    summary = {
        "sparameters": find_items("s-parameters"),
        "mode_frequencies": find_items("mode frequencies"),
        "dispersion_like": find_items("dispersion") + find_items("phase shift") + find_items("phase velocity"),
        "coupling_impedance_like": find_items("coupling", "impedance")
        + find_items("interaction", "impedance")
        + find_items("kc"),
        "wave_impedance_like": find_items("wave impedance"),
        "all_tree_items_count": len(tree_items),
    }
    summary["dsg_bo_required_complete"] = bool(
        summary["sparameters"] and summary["dispersion_like"] and summary["coupling_impedance_like"]
    )
    return summary


def export_manifest(tree_items: list[str], run_dir: Path, parameter_payload: dict[str, Any]) -> None:
    """把结果树清单与参数快照写出到当前 case 目录。"""

    write_text(run_dir / "tree_items.txt", "\n".join(tree_items) + "\n")
    write_json(run_dir / "project_parameters.json", parameter_payload)


def load_result_tree_map(path: Path | None) -> dict[str, str]:
    """读取用户提供的结果树映射 JSON。"""

    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"结果树映射必须是 JSON 对象: {path}")
    return {str(key): str(value) for key, value in payload.items()}


def print_parameter_catalog(parameters: dict[str, ProjectParameter]) -> None:
    """把识别到的参数目录打印到终端。"""

    print("[info] 当前工程识别到的参数:")
    for name, parameter in parameters.items():
        value_text = parameter.value if parameter.value is not None else parameter.expr or ""
        descr_text = parameter.descr or ""
        print(f"  - {name:<16} value={value_text:<12} source={parameter.source} descr={descr_text}")


def run_single_case(
    *,
    cst_interface: Any,
    cst_results: Any,
    project_path: Path,
    case: ParameterCase,
    case_dir: Path,
    poll_seconds: float,
    skip_solver: bool,
    result_tree_map: dict[str, str],
) -> dict[str, Any]:
    """执行单个参数 case 的完整 CST 导出流程。"""

    design_environment = None
    project = None

    try:
        design_environment, project = open_project(cst_interface, project_path)
        apply_parameters_to_project(project, case.parameter_values)
        safe_project_save(project, project_path)

        solver_summary: dict[str, Any] = {"solver_invoked": not skip_solver}
        if not skip_solver:
            solver_summary.update(run_solver(project, poll_seconds))
            safe_project_save(project, project_path)

        parameter_payload = read_parameter_json(project_path)
        result_project = cst_results.ProjectFile(str(project_path), allow_interactive=True)
        results_3d = result_project.get_3d()
        tree_items = get_tree_items(result_project)
        availability = summarize_required_dsg_items(tree_items)

        export_manifest(tree_items, case_dir, parameter_payload)
        exported_1d = export_all_1d_results(results_3d, tree_items, case_dir)
        standard_export_summary = export_standard_dsg_results(
            results_3d=results_3d,
            tree_items=tree_items,
            run_dir=case_dir,
            result_tree_items=result_tree_map,
        )

        if standard_export_summary["required_complete"]:
            status_text = "标准 DSG 结果完整"
        else:
            status_text = f"标准 DSG 结果不完整: {standard_export_summary['missing']}"
        write_text(case_dir / "case_status.txt", status_text + "\n")

        write_json(
            case_dir / "export_summary.json",
            {
                "case_name": case.case_name,
                "project_path": str(project_path),
                "case_dir": str(case_dir),
                "parameter_values": case.parameter_values,
                "changed_parameters": case.changed_parameters,
                "solver_summary": solver_summary,
                "availability": availability,
                "standard_dsg_exports": standard_export_summary,
                "all_1d_exports_count": len(exported_1d),
                "all_1d_exports": exported_1d,
            },
        )

        return {
            "case_name": case.case_name,
            "success": True,
            "failure_reason": "",
            "case_dir": str(case_dir),
            "solver_name": solver_summary.get("solver_name", ""),
            "run_seconds": float(solver_summary.get("run_seconds", 0.0)),
            "changed_parameters": case.changed_parameters,
            "required_complete": bool(standard_export_summary["required_complete"]),
        }
    except Exception as exc:
        traceback_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        write_text(case_dir / "automation_error.txt", traceback_text)
        return {
            "case_name": case.case_name,
            "success": False,
            "failure_reason": str(exc),
            "case_dir": str(case_dir),
            "solver_name": "",
            "run_seconds": 0.0,
            "changed_parameters": case.changed_parameters,
            "required_complete": False,
        }
    finally:
        close_quietly(project, design_environment)


def restore_baseline_parameters(
    *,
    cst_interface: Any,
    project_path: Path,
    baseline_parameters: dict[str, ProjectParameter],
) -> None:
    """把工程参数恢复到脚本启动时识别到的初始值。"""

    design_environment = None
    project = None
    try:
        restore_values = {
            name: parameter_to_text(parameter.value if parameter.value is not None else parameter.expr or "")
            for name, parameter in baseline_parameters.items()
        }
        design_environment, project = open_project(cst_interface, project_path)
        apply_parameters_to_project(project, restore_values)
        safe_project_save(project, project_path)
    finally:
        close_quietly(project, design_environment)


def write_batch_summary(run_dir: Path, summaries: list[dict[str, Any]]) -> None:
    """把所有 case 的运行摘要写成 JSON 和 CSV。"""

    write_json(run_dir / "batch_summary.json", summaries)

    csv_path = run_dir / "batch_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case_name",
                "success",
                "failure_reason",
                "run_seconds",
                "solver_name",
                "required_complete",
                "case_dir",
                "changed_parameters_json",
            ]
        )
        for summary in summaries:
            writer.writerow(
                [
                    summary["case_name"],
                    summary["success"],
                    summary["failure_reason"],
                    summary["run_seconds"],
                    summary["solver_name"],
                    summary["required_complete"],
                    summary["case_dir"],
                    json.dumps(summary["changed_parameters"], ensure_ascii=False),
                ]
            )


def run(args: argparse.Namespace) -> int:
    """执行整条自动化流程。"""

    project_path = args.project_path.resolve()
    if not project_path.exists():
        print(f"[error] 未找到工程文件: {project_path}")
        return 1

    cst_interface, cst_results = load_cst_modules(args.cst_amd64)
    baseline_parameters = discover_project_parameters(cst_interface, project_path)
    fixed_overrides = parse_fixed_overrides(args.set_param)
    sweep_map = parse_sweep_specs(args.sweep)
    explicit_case_rows = load_case_file(args.case_file)
    cases = build_case_definitions(
        baseline_parameters=baseline_parameters,
        fixed_overrides=fixed_overrides,
        sweep_map=sweep_map,
        explicit_case_rows=explicit_case_rows,
    )
    validate_case_parameters(cases, baseline_parameters)

    if args.list_parameters:
        print_parameter_catalog(baseline_parameters)

    run_dir = (args.output_root / "dry_run_preview") if args.dry_run else make_run_dir(args.output_root)
    run_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        run_dir / "parameter_catalog.json",
        {
            name: {
                "expr": parameter.expr,
                "value": parameter.value,
                "descr": parameter.descr,
                "source": parameter.source,
            }
            for name, parameter in baseline_parameters.items()
        },
    )
    write_json(
        run_dir / "planned_cases.json",
        [
            {
                "case_name": case.case_name,
                "parameter_values": case.parameter_values,
                "changed_parameters": case.changed_parameters,
            }
            for case in cases
        ],
    )

    print(f"[info] 当前共计划运行 {len(cases)} 个 case")
    for case in cases:
        print(f"[info] {case.case_name}: {json.dumps(case.changed_parameters, ensure_ascii=False)}")

    if args.dry_run:
        write_json(
            run_dir / "dry_run_summary.json",
            {
                "project_path": str(project_path),
                "parameter_count": len(baseline_parameters),
                "case_count": len(cases),
                "skip_solver": args.skip_solver,
                "list_parameters": args.list_parameters,
            },
        )
        return 0

    result_tree_map = load_result_tree_map(args.result_tree_map)
    summaries: list[dict[str, Any]] = []

    try:
        for index, case in enumerate(cases, start=1):
            case_dir = run_dir / f"{index:03d}_{case.case_name}"
            case_dir.mkdir(parents=True, exist_ok=True)
            write_json(
                case_dir / "input_parameters.json",
                {
                    "case_name": case.case_name,
                    "parameter_values": case.parameter_values,
                    "changed_parameters": case.changed_parameters,
                },
            )
            print(f"[info] 开始运行 {case.case_name} ({index}/{len(cases)})")
            summary = run_single_case(
                cst_interface=cst_interface,
                cst_results=cst_results,
                project_path=project_path,
                case=case,
                case_dir=case_dir,
                poll_seconds=args.poll_seconds,
                skip_solver=args.skip_solver,
                result_tree_map=result_tree_map,
            )
            summaries.append(summary)
            if summary["success"]:
                print(
                    f"[info] {case.case_name} 完成，solver={summary['solver_name']}，"
                    f"耗时 {summary['run_seconds']:.2f} s"
                )
            else:
                print(f"[warn] {case.case_name} 失败: {summary['failure_reason']}")
    finally:
        if not args.keep_last_parameters:
            print("[info] 正在把 CST 工程参数恢复到初始基线...")
            restore_baseline_parameters(
                cst_interface=cst_interface,
                project_path=project_path,
                baseline_parameters=baseline_parameters,
            )

    write_batch_summary(run_dir, summaries)
    success_count = sum(1 for item in summaries if item["success"])
    print(f"[info] 批量运行结束：成功 {success_count}/{len(summaries)}")
    return 0 if success_count == len(summaries) else 1


def main() -> int:
    """脚本主入口。"""

    args = parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
