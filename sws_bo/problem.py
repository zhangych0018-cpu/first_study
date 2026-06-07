"""本模块提供项目当前默认问题定义的统一入口，使脚本层无需关心具体采用的是哪个慢波结构问题类。当前项目已经 DSG-only，因此这里实质上转发到 DSG 问题定义。"""

from __future__ import annotations

from .problems.dsg_bwo_problem import DSGSWSProblem


__all__ = ["DSGSWSProblem"]
