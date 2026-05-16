"""
Task protocol (Phase 3).

A `Task` owns everything specific to one objective: how to build its
head, what it reports as metrics, and the forward math (encoder call →
head → loss/logits/targets). This replaces the `objectives.compute_*`
functions, which reached back into `pl_module` for heads, metrics,
`.log`, `.device`, and `.config`.

The forward signature is `forward(model, batch) -> TaskOutput`. `model`
is anything exposing the small surface tasks need:

    model.backbone : Backbone
    model.heads    : Mapping[str, nn.Module]   (this task's head at model.heads[self.name])
    model.config   : dict
    model.device   : torch.device

`RenaissanceModel` (Phase 4) provides exactly this. Phase 3 tests use a
lightweight stand-in with the same attributes.

Metric *aggregation* (torchmetrics objects, train/dev/test routing,
TensorBoard flush) is deliberately NOT here — that was the worst of the
`pl_module` coupling. Tasks return raw `loss` / `logits` / `targets`;
`metric_names()` declares what the model should track. Phase 4/6 wire the
metric objects and the trainer flush.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..backbones import Backbone


@dataclass
class TaskOutput:
    """What every task returns.

    loss
        Scalar training loss.
    logits
        Raw model outputs (for metric computation downstream). May be
        None for tasks without a natural logit (none currently).
    targets
        The ground truth the metrics compare against (label ids, soft
        targets, bbox tensor, ...).
    extras
        Task-specific carry-through (e.g. VQA per-sample label/score
        lists the VQA-score metric needs).
    """

    loss: torch.Tensor
    logits: Optional[torch.Tensor] = None
    targets: Optional[torch.Tensor] = None
    extras: dict = field(default_factory=dict)


class Task(ABC):
    """One objective. Stateless singleton — all parameters live in the
    head, which the model owns. `name` is the registry key and the
    `model.heads` key."""

    name: str

    def build_head(self, backbone: Backbone, config: Dict) -> Optional[nn.Module]:
        """Construct this task's head. `backbone` supplies dims
        (`pooled_dim`, `token_dim`, `text_hidden_size`, ...); `config`
        supplies task params (`vocab_size`, `vqav2_label_size`, ...).
        Return None for a head-less task."""
        return None

    def metric_names(self) -> Tuple[str, ...]:
        """Metrics this task reports, beyond loss. Phase 4 maps these to
        torchmetrics objects."""
        return ()

    @abstractmethod
    def forward(self, model, batch) -> TaskOutput:
        ...
