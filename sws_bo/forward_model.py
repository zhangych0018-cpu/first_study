"""Abstract forward simulator interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np
import torch

from .data_schema import SimulationResult


class ForwardSimulator(ABC):
    """Abstract base class for CST-like simulators."""

    @abstractmethod
    def run(self, x: dict | np.ndarray | torch.Tensor) -> SimulationResult:
        """Evaluate one design point."""

    def run_batch(self, xs: Iterable[dict | np.ndarray | torch.Tensor]) -> list[SimulationResult]:
        """Evaluate a batch sequentially."""

        return [self.run(x) for x in xs]
