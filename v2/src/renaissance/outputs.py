"""Shared output contracts between encoders, fusion modules, and tasks."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class TokenStream:
    """What an encoder produces: a token sequence plus its attention mask."""

    tokens: torch.Tensor  # (B, N, hidden)
    mask: torch.Tensor | None = None  # (B, N); None means all tokens attend


@dataclass
class FusionOutput:
    """Uniform fusion contract (1.3's EncoderOutput, minus the legacy aliases)."""

    pooled: torch.Tensor  # (B, pooled_dim)
    text_tokens: torch.Tensor  # (B, T, token_dim)
    image_tokens: torch.Tensor  # (B, N_img, token_dim)
    text_ids: torch.Tensor | None = None
    text_labels: torch.Tensor | None = None
    text_masks: torch.Tensor | None = None


@dataclass
class TaskOutput:
    """What a task's forward returns; metric aggregation happens outside the task."""

    loss: torch.Tensor
    logits: torch.Tensor | None = None
    targets: torch.Tensor | None = None
    extras: dict = field(default_factory=dict)
