"""本模块定义项目内部统一使用的数据结构，例如仿真结果、优化记录和分析阶段需要的结构化对象。它的目标是减少各模块之间的数据格式歧义。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SimulationResult:
    """描述任意前向仿真后端返回的标准结果结构，统一承载目标值、约束值、运行状态和附加元数据。"""

    Kc_mean: float
    vp_std: float
    ohmic_loss_mean: float
    S11_max: float
    S21_mean: float | None = None
    bandwidth_estimate: float | None = None
    sim_time: float = 0.0
    success: bool = True
    failure_reason: str | None = None
    raw_files: dict[str, str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    extra_outputs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """把结果对象转换成适合 JSON 或 DataFrame 序列化的普通字典，方便写盘与日志记录。"""

        payload = asdict(self)
        extra = payload.pop("extra_outputs", {}) or {}
        payload.update(extra)
        return payload


@dataclass
class CandidateRecommendation:
    """保存采集函数推荐的候选点，并同时记录归一化坐标与物理参数坐标，方便优化和导出复用。"""

    normalized_x: list[float]
    physical_x: list[float]
    score: float
    feasibility_probability: float | None = None


@dataclass
class RobustEvaluation:
    """描述在制造扰动下的鲁棒评估摘要，统一保存名义性能、扰动统计量和风险度量结果。"""

    nominal_objective: list[float]
    expected_objective: list[float]
    worst_case_objective: list[float]
    cvar_objective: list[float]
    feasibility_rate: float
