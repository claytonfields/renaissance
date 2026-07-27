"""Encoder protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from torch import nn

from renaissance.outputs import TokenStream


class Encoder(nn.Module, ABC):
    """A single-modality tower producing one token stream per input.

    Registered classes must own a ``Config`` dataclass and are constructed
    with it: ``EncoderCls(cfg)``.
    """

    Config: ClassVar[type]

    @abstractmethod
    def forward(self, batch: dict) -> TokenStream | list[TokenStream]:
        """Encode the batch. Image encoders return one TokenStream per image slot."""

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """Width of the produced token embeddings."""
