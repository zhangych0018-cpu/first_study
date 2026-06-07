"""本模块定义 DSG 慢波结构的几何配置对象和参数展开逻辑，用于把优化变量转换为工程几何描述与 CST 参数字典。它是结构替换后的几何基础层。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..problems.dsg_bwo_problem import DSGSWSProblem


@dataclass
class DSGGeometryConfig:
    """封装 DSG 慢波结构中不参与优化但需要长期固定的几何或物理参数配置。"""

    N_periods: int = 80
    target_frequency_ghz: float = 100.0
    working_band_ghz: tuple[float, float] = (96.0, 110.0)
    target_mode: str = "TM21_like"
    conductivity: float = 2.25e7
    rounding_radius: float = 0.0
    beam_width: float = 2.8


@dataclass
class DSGGeometryValidation:
    """保存几何合法性检查的结果，包括是否通过、失败原因以及工程性软告警。"""

    is_valid: bool
    warnings: list[str] = field(default_factory=list)


class DSGGeometryBuilder:
    """负责把优化变量扩展为单周期或有限长 DSG 结构所需的完整参数字典。"""

    def __init__(self, config: DSGGeometryConfig | None = None) -> None:
        self.config = config or DSGGeometryConfig()

    def build_unit_cell_parameters(self, x: dict | np.ndarray) -> dict[str, float]:
        return build_unit_cell_parameters(x, config=self.config)

    def build_finite_length_parameters(self, x: dict | np.ndarray) -> dict[str, float]:
        return build_finite_length_parameters(x, config=self.config)

    def export_cst_parameter_dict(self, x: dict | np.ndarray) -> dict[str, float]:
        return export_cst_parameter_dict(x, config=self.config)

    def summarize_geometry(self, x: dict | np.ndarray) -> dict[str, float | list[str]]:
        return summarize_geometry(x, config=self.config)


def _to_param_dict(x: dict | np.ndarray) -> dict[str, float]:
    if isinstance(x, dict):
        return {name: float(x[name]) for name in DSGSWSProblem.param_names}
    return DSGSWSProblem.as_dict(np.asarray(x, dtype=float))


def validate_dsg_geometry(x: dict | np.ndarray) -> DSGGeometryValidation:
    """对 DSG 结构参数执行工程代理检查，并给出软告警而不是一味硬失败。"""

    params = _to_param_dict(x)
    warnings: list[str] = []

    in_bounds = DSGSWSProblem.validate_x(np.array([params[name] for name in DSGSWSProblem.param_names], dtype=float))
    hard_valid = (
        in_bounds
        and params["T"] > 0.0
        and params["G"] > 0.0
        and params["H"] > 0.0
        and params["G"] < params["P"]
    )
    if not hard_valid:
        return DSGGeometryValidation(is_valid=False, warnings=warnings)

    if params["H"] > 0.68:
        warnings.append("structure_height_high")
    if params["T"] < 0.25:
        warnings.append("beam_tunnel_narrow")
    if params["W"] < 3.0 or (params["H"] / max(params["T"], 1e-9)) > 2.4:
        warnings.append("mode_competition_risk")
    if params["G"] < 0.15 or params["G"] > 0.28 or params["T"] < 0.24:
        warnings.append("manufacturing_risk")

    return DSGGeometryValidation(is_valid=True, warnings=warnings)


def build_unit_cell_parameters(x: dict | np.ndarray, config: DSGGeometryConfig | None = None) -> dict[str, float]:
    """构造单周期建模所需的参数集合，包括若干派生几何量，方便周期单元分析。"""

    cfg = config or DSGGeometryConfig()
    params = _to_param_dict(x)
    params.update(
        {
            "period_shift": 0.5 * params["P"],
            "beam_half_height": 0.5 * params["T"],
            "upper_grating_y_min": 0.5 * params["T"],
            "upper_grating_y_max": 0.5 * params["T"] + params["H"],
            "lower_grating_y_min": -0.5 * params["T"] - params["H"],
            "lower_grating_y_max": -0.5 * params["T"],
            "beam_width": cfg.beam_width,
            "conductivity": cfg.conductivity,
            "rounding_radius": cfg.rounding_radius,
            "target_frequency_ghz": cfg.target_frequency_ghz,
        }
    )
    return params


def build_finite_length_parameters(x: dict | np.ndarray, config: DSGGeometryConfig | None = None) -> dict[str, float]:
    """构造有限长慢波结构所需的参数集合，使端口驱动或 cold-test 模型也能复用同一套变量。"""

    cfg = config or DSGGeometryConfig()
    unit = build_unit_cell_parameters(x, config=cfg)
    unit.update(
        {
            "N_periods": cfg.N_periods,
            "total_length_mm": cfg.N_periods * unit["P"],
            "working_band_low_ghz": cfg.working_band_ghz[0],
            "working_band_high_ghz": cfg.working_band_ghz[1],
            "target_mode": cfg.target_mode,
        }
    )
    return unit


def export_cst_parameter_dict(x: dict | np.ndarray, config: DSGGeometryConfig | None = None) -> dict[str, float]:
    """把几何参数整理成适合直接写入 CST 工程参数表的字典格式。"""

    params = build_finite_length_parameters(x, config=config)
    ordered_keys = ["W", "P", "T", "G", "H", "N_periods", "conductivity", "rounding_radius"]
    return {key: float(params[key]) for key in ordered_keys}


def summarize_geometry(x: dict | np.ndarray, config: DSGGeometryConfig | None = None) -> dict[str, float | list[str]]:
    """输出带有派生量和告警信息的几何摘要，便于调试参数、检查建模合理性和记录实验设置。"""

    cfg = config or DSGGeometryConfig()
    params = build_finite_length_parameters(x, config=cfg)
    validation = validate_dsg_geometry(x)
    summary: dict[str, float | list[str]] = {
        **params,
        "aspect_ratio_H_over_T": float(params["H"] / max(params["T"], 1e-9)),
        "duty_cycle_G_over_P": float(params["G"] / max(params["P"], 1e-9)),
        "is_valid": float(validation.is_valid),
        "warnings": validation.warnings,
    }
    return summary
