"""The composed model: encoders + fusion + heads, Hub-native."""

from __future__ import annotations

from huggingface_hub import PyTorchModelHubMixin
from torch import nn

from renaissance import registry
from renaissance.model_config import ModelConfig
from renaissance.outputs import FusionOutput


class RenaissanceModel(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="renaissance",
    coders={ModelConfig: (lambda c: c.to_dict(), lambda d: ModelConfig.from_dict(d))},
):
    """Composition, not a class hierarchy: two-tower ≡ two encoders +
    cross-attention fusion. ``save_pretrained`` / ``from_pretrained`` /
    ``push_to_hub`` come from PyTorchModelHubMixin (safetensors + config.json).
    """

    def __init__(self, config: ModelConfig | dict):
        super().__init__()
        # from_pretrained hands back the raw config.json dict (the mixin's
        # coders decode only fires on resolvable annotations).
        if isinstance(config, dict):
            config = ModelConfig.from_dict(config)
        self.config = config

        built = {}
        for key, spec in config.encoders.items():
            enc_cls = registry.get("encoder", spec.name)
            built[key] = enc_cls(enc_cls.Config(**spec.args))
        self.encoders = nn.ModuleDict(built)

        fusion_cls = registry.get("fusion", config.fusion.name)
        self.fusion = fusion_cls(
            fusion_cls.Config(**config.fusion.args),
            text_dim=self.encoders["text"].hidden_size,
            image_dim=self.encoders["image"].hidden_size,
        )

        self.heads = nn.ModuleDict()  # populated per-task from Phase 4

    def encode(self, batch: dict) -> FusionOutput:
        text = self.encoders["text"](batch)
        images = self.encoders["image"](batch)
        return self.fusion(text, images)

    def forward(self, batch: dict) -> FusionOutput:
        # No tasks yet (Phase 1): forward is the fusion output, like 1.3's
        # no-active-tasks infer path.
        return self.encode(batch)

    @property
    def pooled_dim(self) -> int:
        return self.fusion.pooled_dim

    @property
    def token_dim(self) -> int:
        return self.fusion.token_dim
