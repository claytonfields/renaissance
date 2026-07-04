"""
The one place HuggingFace encoders get loaded.

Replaces the scattered `AutoConfig`/`AutoModel` + `*_manual_configuration`
+ try/except-cascade logic that was duplicated across the legacy
one-tower and two-tower encoders. Two functions:

- `load_hf_encoder(name, ...)` — random-init-from-config vs download
  pretrained, optional dim overrides, optional freeze.
- `hf_model_hidden_size(model)` — robust last-stage hidden size across
  Bert/ViT/Electra/DeiT/ConvNeXt/Swin-style configs.
"""

from typing import Dict, Optional, Tuple

import torch.nn as nn
from transformers import AutoConfig, AutoModel


def hf_model_hidden_size(model: nn.Module) -> int:
    """Return the encoder's output hidden size.

    HF configs disagree on the attribute name. Plain transformers expose a
    scalar ``hidden_size`` (Bert/ViT/Electra/DeiT) or ``dim``; pyramid
    vision models expose a per-stage list under ``hidden_sizes`` /
    ``embed_dims`` / ``embed_dim`` and we want the last (deepest) stage.
    """
    cfg = model.config
    for attr in ("hidden_size", "dim"):
        v = getattr(cfg, attr, None)
        if isinstance(v, int):
            return v
    for attr in ("hidden_sizes", "embed_dims", "embed_dim"):
        v = getattr(cfg, attr, None)
        if isinstance(v, (list, tuple)) and v:
            return v[-1]
        if isinstance(v, int):
            return v
    raise ValueError(
        f"Cannot determine hidden size from {type(cfg).__name__}; "
        "extend hf_model_hidden_size for this architecture."
    )


def load_hf_encoder(
    name: str,
    *,
    random_init: bool = True,
    overrides: Optional[Dict] = None,
    freeze: bool = False,
    attn_implementation: Optional[str] = None,
) -> Tuple[nn.Module, int]:
    """Build or download an HF encoder.

    Parameters
    ----------
    name
        HF Hub model id or local path.
    random_init
        If True, build from config only (no weight download). If False,
        download pretrained weights.
    overrides
        AutoConfig kwargs (e.g. ``hidden_size``, ``num_hidden_layers``)
        injected only when ``random_init=True`` — pretrained weights pin
        the dims, so overriding them is a silent footgun and is rejected.
    freeze
        If True, set ``requires_grad=False`` on every parameter.
    attn_implementation
        Optional attention backend to select at build time (e.g.
        ``"flash_attention_2"``, ``"sdpa"``, ``"eager"``). Forwarded to
        ``AutoModel.from_pretrained`` / ``from_config``; HF's own
        validation raises if the backend isn't available for this model
        or the env. Leave ``None`` (default) for HF's default choice.

    Returns
    -------
    (model, hidden_size)
    """
    extra: Dict = {}
    if attn_implementation is not None:
        extra["attn_implementation"] = attn_implementation

    if random_init:
        config = AutoConfig.from_pretrained(name, **(overrides or {}))
        model = AutoModel.from_config(config, **extra)
    else:
        if overrides:
            raise ValueError(
                "overrides only apply when random_init=True; pretrained "
                "weights fix the architecture dims."
            )
        model = AutoModel.from_pretrained(name, **extra)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    return model, hf_model_hidden_size(model)
