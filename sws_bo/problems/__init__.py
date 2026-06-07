"""本模块是问题定义子包的入口，负责统一暴露当前工程使用的 DSG 问题类。脚本和优化器可通过这里获得一致的问题接口。"""

from __future__ import annotations

from .dsg_bwo_problem import DSGSParameterProblem, DSGSWSProblem


PROBLEM_REGISTRY: dict[str, type] = {
    "default": DSGSWSProblem,
    "dsg": DSGSWSProblem,
    "dsg_bwo": DSGSWSProblem,
    "dsg_w_band_bwo": DSGSWSProblem,
    "dsg_sparameter": DSGSParameterProblem,
    "dsg_sparameters": DSGSParameterProblem,
    "dsg_w_band_sparameter": DSGSParameterProblem,
}


def resolve_problem(problem: str | type | None) -> type:
    """把问题标识解析为具体的问题定义类，供脚本层和优化器以统一方式获取当前使用的问题对象。"""

    if problem is None:
        return DSGSWSProblem
    if isinstance(problem, str):
        key = problem.strip().lower()
        if key not in PROBLEM_REGISTRY:
            raise KeyError(f"Unknown problem '{problem}'. Available: {sorted(PROBLEM_REGISTRY)}")
        return PROBLEM_REGISTRY[key]
    return problem


__all__ = ["DSGSParameterProblem", "DSGSWSProblem", "PROBLEM_REGISTRY", "resolve_problem"]
