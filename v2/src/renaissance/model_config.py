"""Model composition config — the payload serialized into config.json."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class EncoderSpec:
    """Which registered encoder to build, and its component-config kwargs."""

    name: str
    args: dict = field(default_factory=dict)


@dataclass
class FusionSpec:
    """Which registered fusion to build, and its component-config kwargs."""

    name: str = "cross-attention"
    args: dict = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Composition of a RenaissanceModel: encoders + fusion (+ tasks, Phase 4)."""

    encoders: dict[str, EncoderSpec]  # keys are modality slots, e.g. "text", "image"
    fusion: FusionSpec
    tasks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ModelConfig:
        return cls(
            encoders={key: EncoderSpec(**spec) for key, spec in d["encoders"].items()},
            fusion=FusionSpec(**d["fusion"]),
            tasks=list(d.get("tasks", [])),
        )
