"""Task protocol — a task is one self-contained vertical slice.

Each task declares its own config, batch requirements, head, collate,
forward (loss), and metrics. Implementations land in Phase 4 (mlm, itm)
and Phase 6 (classification tasks, NLVR2 as a plain 2-image task).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

from torch import nn

from renaissance.outputs import TaskOutput


@dataclass
class BatchSpec:
    """What the data layer must produce for a task — drives the generic loader."""

    n_images: int = 1
    text: bool = True
    label: str | None = None  # e.g. "binary", "multiclass", None for self-supervised


class Task(ABC):
    Config: ClassVar[type]
    name: ClassVar[str]
    batch_spec: ClassVar[BatchSpec]

    def build_head(self, fusion_dim: int) -> nn.Module | None:
        """Construct this task's head (None for headless tasks)."""
        return None

    @abstractmethod
    def forward(self, model: Any, batch: dict) -> TaskOutput:
        """Compute the task loss (and logits/targets for metrics)."""

    def collate(self, examples: list, processors: dict) -> dict:
        """Example→tensor conversion using checkpoint-derived processors (Phase 3)."""
        raise NotImplementedError(f"{type(self).__name__}.collate lands with the data layer (Phase 3)")

    def metrics(self) -> list[str]:
        return []
