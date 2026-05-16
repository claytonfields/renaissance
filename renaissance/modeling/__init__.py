"""
Modern modeling layer for Renaissance.

Phased rewrite of `renaissance.modules`. New code lives here and grows
incrementally; the legacy package stays importable until the final
cutover. See the project memory for the phase plan.

Phase 1 status: backbones only. `RenaissanceModel`, tasks, and the wiring
into `run.py` arrive in later phases.
"""

from .backbones import (
    Backbone,
    EncoderOutput,
    OneTowerBackbone,
    TwoTowerBackbone,
    build_backbone,
)
from .model import RenaissanceModel
from .tasks import TASK_REGISTRY, Task, TaskOutput, get_task

__all__ = [
    "Backbone",
    "EncoderOutput",
    "OneTowerBackbone",
    "TwoTowerBackbone",
    "build_backbone",
    "Task",
    "TaskOutput",
    "TASK_REGISTRY",
    "get_task",
    "RenaissanceModel",
]
