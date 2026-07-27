"""HuggingFace-backed text and image encoders."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from renaissance.components.encoders.base import Encoder
from renaissance.components.encoders.hf_loader import load_hf_encoder
from renaissance.outputs import TokenStream
from renaissance.registry import register


@dataclass
class HFEncoderConfig:
    """Component-owned config shared by the hf-text and hf-image encoders."""

    hub_id: str = ""
    random_init: bool = False  # True = build from AutoConfig only, no weight download
    overrides: dict = field(default_factory=dict)  # AutoConfig kwargs; random_init only
    freeze: bool = False
    attn_implementation: str | None = None


class _HFEncoder(Encoder):
    Config = HFEncoderConfig

    def __init__(self, cfg: HFEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.model, self._hidden_size = load_hf_encoder(
            cfg.hub_id,
            random_init=cfg.random_init,
            overrides=cfg.overrides or None,
            freeze=cfg.freeze,
            attn_implementation=cfg.attn_implementation,
        )

    @property
    def hidden_size(self) -> int:
        return self._hidden_size


@register("encoder", "hf-text")
class HFTextEncoder(_HFEncoder):
    """Text tower. Reads ``text_ids`` / ``text_masks`` from the batch."""

    def forward(self, batch: dict) -> TokenStream:
        tokens = self.model(
            input_ids=batch["text_ids"], attention_mask=batch["text_masks"]
        ).last_hidden_state
        return TokenStream(tokens=tokens, mask=batch["text_masks"])


@register("encoder", "hf-image")
class HFImageEncoder(_HFEncoder):
    """Image tower. Reads ``images`` — (B, K, C, H, W), or (B, C, H, W) as K=1 —
    and returns one TokenStream per image slot."""

    def forward(self, batch: dict) -> list[TokenStream]:
        images = batch["images"]
        if images.dim() == 4:
            images = images.unsqueeze(1)
        if images.dim() != 5:
            raise ValueError(f"images must be (B, K, C, H, W) or (B, C, H, W); got {tuple(images.shape)}")

        streams = []
        for k in range(images.size(1)):
            streams.append(TokenStream(tokens=self._encode_one(images[:, k])))
        return streams

    def _encode_one(self, img: torch.Tensor) -> torch.Tensor:
        try:
            out = self.model(img, interpolate_pos_encoding=True)
        except TypeError:  # encoder doesn't support pos-encoding interpolation
            out = self.model(img)
        tokens = out.last_hidden_state
        if tokens.dim() == 4:
            raise NotImplementedError(
                "Conv-stage vision encoders (ConvNeXt/Swin-style 4D outputs) are "
                "deferred; see 1.3's resize_convolutional_output for the port source."
            )
        return tokens
