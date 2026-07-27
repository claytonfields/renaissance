"""Bi-directional cross-attention fusion (LXMERT X-layers).

Ported from the 1.3 two-tower encoder (`two_tower_encoder.py`): input
projections into the fusion width, token-type embeddings, a stack of
``LxmertXLayer``s, and two CLS poolers whose concat is the pooled output
(``pooled_dim = 2 * hidden_size``).

The token-type table is sized ``1 + max_images`` from config, so growing
to multi-image tasks (NLVR2, Phase 6) is a config choice — this replaces
1.3's ``adjust_type_embeds_for_nlvr2`` weight surgery.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn
from transformers.models.lxmert.configuration_lxmert import LxmertConfig
from transformers.models.lxmert.modeling_lxmert import LxmertXLayer

from renaissance.components.fusion.base import FusionModule
from renaissance.components.heads.pooler import Pooler, init_weights
from renaissance.outputs import FusionOutput, TokenStream
from renaissance.registry import register


@dataclass
class CrossAttentionConfig:
    hidden_size: int = 384
    num_layers: int = 6
    num_heads: int = 6
    mlp_ratio: int = 4
    drop_rate: float = 0.0
    max_images: int = 1  # sizes the token-type table: text is type 0, image k is type 1+k


@register("fusion", "cross-attention")
class CrossAttentionFusion(FusionModule):
    Config = CrossAttentionConfig

    def __init__(self, cfg: CrossAttentionConfig, *, text_dim: int, image_dim: int):
        super().__init__()
        if cfg.hidden_size % cfg.num_heads:
            raise ValueError(f"hidden_size {cfg.hidden_size} must be divisible by num_heads {cfg.num_heads}")
        self.cfg = cfg

        self.text_transform = nn.Linear(text_dim, cfg.hidden_size)
        self.image_transform = nn.Linear(image_dim, cfg.hidden_size)
        self.token_type_embeddings = nn.Embedding(1 + cfg.max_images, cfg.hidden_size)

        # LxmertXLayer reads only the attention/FFN fields of the config —
        # vocab/position sizes belong to the (unused) LXMERT embedding stack.
        lxmert_config = LxmertConfig(
            hidden_size=cfg.hidden_size,
            num_attention_heads=cfg.num_heads,
            intermediate_size=cfg.hidden_size * cfg.mlp_ratio,
            hidden_dropout_prob=cfg.drop_rate,
            attention_probs_dropout_prob=cfg.drop_rate,
        )
        self.layers = nn.ModuleList(LxmertXLayer(lxmert_config) for _ in range(cfg.num_layers))
        self.text_pooler = Pooler(cfg.hidden_size)
        self.image_pooler = Pooler(cfg.hidden_size)

        self.apply(init_weights)

    @property
    def pooled_dim(self) -> int:
        return 2 * self.cfg.hidden_size

    @property
    def token_dim(self) -> int:
        return self.cfg.hidden_size

    def forward(self, text: TokenStream, images: Sequence[TokenStream]) -> FusionOutput:
        if len(images) != 1:
            raise NotImplementedError(
                f"cross-attention fusion currently supports exactly one image stream "
                f"(got {len(images)}); multi-image fusion lands with NLVR2 (Phase 6)"
            )
        image = images[0]

        text_embeds = self.text_transform(text.tokens)
        image_embeds = self.image_transform(image.tokens)

        text_mask = self._mask_or_ones(text, text_embeds)
        image_mask = self._mask_or_ones(image, image_embeds)

        text_embeds = text_embeds + self.token_type_embeddings(torch.zeros_like(text_mask))
        image_embeds = image_embeds + self.token_type_embeddings(torch.ones_like(image_mask))

        ext_text_mask = self._extend_mask(text_mask, text_embeds.dtype)
        ext_image_mask = self._extend_mask(image_mask, image_embeds.dtype)

        for layer in self.layers:
            text_embeds, image_embeds = layer(text_embeds, ext_text_mask, image_embeds, ext_image_mask)[:2]

        pooled = torch.cat([self.text_pooler(text_embeds), self.image_pooler(image_embeds)], dim=-1)
        return FusionOutput(pooled=pooled, text_tokens=text_embeds, image_tokens=image_embeds, text_masks=text.mask)

    @staticmethod
    def _mask_or_ones(stream: TokenStream, embeds: torch.Tensor) -> torch.Tensor:
        if stream.mask is not None:
            return stream.mask.long()
        return torch.ones(embeds.shape[:2], dtype=torch.long, device=embeds.device)

    @staticmethod
    def _extend_mask(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """(B, N) 0/1 mask -> (B, 1, 1, N) additive attention mask."""
        return (1.0 - mask[:, None, None, :].to(dtype)) * torch.finfo(dtype).min
