"""本模块是 sws_bo 包的顶层入口，用于暴露项目级公共对象、版本信息或便捷导入路径。它帮助外部脚本以更简洁的方式访问 DSG 优化框架。"""

from .data_schema import SimulationResult
from .problems.dsg_bwo_problem import DSGSWSProblem

__all__ = ["DSGSWSProblem", "SimulationResult"]
