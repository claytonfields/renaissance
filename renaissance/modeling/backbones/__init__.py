"""Backbone protocol, implementations, and a config-driven factory."""

from .base import Backbone, EncoderOutput
from .hf_loader import hf_model_hidden_size, load_hf_encoder
from .one_tower import OneTowerBackbone
from .two_tower import TwoTowerBackbone


def build_backbone(config) -> Backbone:
    """Instantiate the backbone named by ``config['model_type']``."""
    model_type = config["model_type"]
    if model_type == "one-tower":
        return OneTowerBackbone(config)
    if model_type == "two-tower":
        return TwoTowerBackbone(config)
    raise ValueError(
        f"Unknown model_type {model_type!r}; expected 'one-tower' or 'two-tower'."
    )


__all__ = [
    "Backbone",
    "EncoderOutput",
    "OneTowerBackbone",
    "TwoTowerBackbone",
    "build_backbone",
    "load_hf_encoder",
    "hf_model_hidden_size",
]
