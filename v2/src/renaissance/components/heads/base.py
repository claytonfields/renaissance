"""Head protocol."""

from __future__ import annotations

from abc import ABC
from typing import Any, ClassVar

from torch import nn


class Head(nn.Module, ABC):
    """A task output head: ``HeadCls(cfg, in_dim)`` where in_dim comes from fusion."""

    Config: ClassVar[type]

    def __init__(self, cfg: Any, in_dim: int) -> None:  # noqa: B027  (concrete heads override)
        super().__init__()
