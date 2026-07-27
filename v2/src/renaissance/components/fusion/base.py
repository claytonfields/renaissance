"""Fusion protocol: combine a text stream with one or more image streams."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar

from torch import nn

from renaissance.outputs import FusionOutput, TokenStream


class FusionModule(nn.Module, ABC):
    """Fuses encoder token streams into the uniform FusionOutput contract.

    Constructed as ``FusionCls(cfg, text_dim=..., image_dim=...)`` — the
    composed model injects the encoder widths so fusion configs never
    duplicate them.
    """

    Config: ClassVar[type]

    @abstractmethod
    def __init__(self, cfg: Any, *, text_dim: int, image_dim: int) -> None: ...

    @abstractmethod
    def forward(self, text: TokenStream, images: Sequence[TokenStream]) -> FusionOutput:
        """Fuse one text stream with K image streams (task-declared n_images)."""

    @property
    @abstractmethod
    def pooled_dim(self) -> int:
        """Width of ``FusionOutput.pooled``."""

    @property
    @abstractmethod
    def token_dim(self) -> int:
        """Width of ``FusionOutput.text_tokens`` / ``image_tokens``."""
