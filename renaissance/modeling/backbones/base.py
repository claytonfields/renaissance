"""
Backbone protocol + the uniform output container.

A `Backbone` turns a batch dict into an `EncoderOutput`. One-tower and
two-tower are two implementations of the same interface — task heads and
`RenaissanceModel` (later phases) only ever see this protocol, never the
concrete encoder internals.

`EncoderOutput` exposes new attribute names (`pooled`, `text_tokens`,
`image_tokens`) but also supports the legacy dict keys (`cls_feats`,
`text_feats`, `image_feats`) via `__getitem__`/`__contains__`, so the
existing `objectives.compute_*` code keeps working unchanged until the
task layer is rewritten in Phase 3.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

# Legacy dict-key → new attribute name. Kept until Phase 3 migrates the
# task layer off dict access.
_LEGACY_ALIASES = {
    "cls_feats": "pooled",
    "text_feats": "text_tokens",
    "image_feats": "image_tokens",
}


@dataclass
class EncoderOutput:
    """Uniform encoder result.

    Attributes
    ----------
    pooled
        Joint CLS feature consumed by multimodal heads. Shape
        ``(B, pooled_dim)``.
    text_tokens
        Per-token text features. Shape ``(B, T, text_hidden)``.
    image_tokens
        Per-token / per-patch image features. Shape
        ``(B, N, image_hidden)``.
    text_ids, text_labels, text_masks
        Text bookkeeping the MLM task needs. When the backbone is called
        with text masking on, ``text_ids`` / ``text_labels`` are the
        mlm-substituted tensors.
    """

    pooled: torch.Tensor
    text_tokens: torch.Tensor
    image_tokens: torch.Tensor
    text_ids: Optional[torch.Tensor] = None
    text_labels: Optional[torch.Tensor] = None
    text_masks: Optional[torch.Tensor] = None

    def _resolve(self, key):
        return _LEGACY_ALIASES.get(key, key)

    def __getitem__(self, key):
        return getattr(self, self._resolve(key))

    def __contains__(self, key):
        return getattr(self, self._resolve(key), None) is not None

    def get(self, key, default=None):
        return getattr(self, self._resolve(key), default)


class Backbone(nn.Module, ABC):
    """Protocol every encoder implements.

    `forward` returns an `EncoderOutput`. `pooled_dim` is the width of the
    joint CLS feature heads receive. `text_hidden_size` /
    `image_hidden_size` are the per-stream widths (used by text-only /
    image-only heads); a backbone that doesn't separate streams may raise
    `NotImplementedError`.
    """

    @abstractmethod
    def forward(
        self,
        batch,
        *,
        mask_text: bool = False,
        mask_image: bool = False,
        image_token_type_idx: int = 1,
    ) -> EncoderOutput:
        ...

    @property
    @abstractmethod
    def pooled_dim(self) -> int:
        ...

    @property
    @abstractmethod
    def token_dim(self) -> int:
        """Width of `text_tokens` / `image_tokens` as returned by
        `forward` — i.e. the per-token feature size the MLM head consumes.
        One-tower: the shared encoder hidden size. Two-tower: the fused
        cross-modal width (`cross_layer_hidden_size`), NOT the raw text/
        image encoder widths."""
        ...

    @property
    def text_hidden_size(self) -> int:
        raise NotImplementedError(
            f"{type(self).__name__} does not expose a separate text stream"
        )

    @property
    def image_hidden_size(self) -> int:
        raise NotImplementedError(
            f"{type(self).__name__} does not expose a separate image stream"
        )

    def encode_text_only(self, batch) -> torch.Tensor:
        """Text-only forward for GLUE-style tasks. Returns the text
        encoder's last hidden state, shape ``(B, T, text_hidden)``."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support text-only encoding"
        )
