"""本模块抽象前向仿真接口，统一约束真实 CST、mock 仿真器和未来可能接入的其他求解后端。优化主循环只依赖这里定义的协议，从而保持后端可替换。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np
import torch

from .data_schema import SimulationResult


class ForwardSimulator(ABC):
    """定义 CST 类仿真器的抽象接口，要求所有后端都以统一形式接收设计点并返回标准仿真结果。"""

    @abstractmethod
    def run(self, x: dict | np.ndarray | torch.Tensor) -> SimulationResult:
        """评估单个设计点，并返回该点对应的标准化仿真结果。"""

    def run_batch(self, xs: Iterable[dict | np.ndarray | torch.Tensor]) -> list[SimulationResult]:
        """按顺序评估一批设计点，默认逐点调用单点接口，保证最基础的批处理能力。"""

        return [self.run(x) for x in xs]
