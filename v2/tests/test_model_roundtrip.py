"""save_pretrained / from_pretrained roundtrip via PyTorchModelHubMixin."""

import json

import torch

from renaissance import EncoderSpec, FusionSpec, ModelConfig, RenaissanceModel
from tests.conftest import TINY_FUSION_ARGS, TINY_IMAGE_ARGS, TINY_TEXT_ARGS, make_batch


def _tiny_config() -> ModelConfig:
    return ModelConfig(
        encoders={
            "text": EncoderSpec(name="hf-text", args=dict(TINY_TEXT_ARGS)),
            "image": EncoderSpec(name="hf-image", args=dict(TINY_IMAGE_ARGS)),
        },
        fusion=FusionSpec(name="cross-attention", args=dict(TINY_FUSION_ARGS)),
    )


def test_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = RenaissanceModel(_tiny_config())
    model.eval()

    model.save_pretrained(tmp_path)
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "model.safetensors").exists()

    reloaded = RenaissanceModel.from_pretrained(tmp_path)
    reloaded.eval()

    assert reloaded.config == model.config

    original = dict(model.state_dict())
    restored = dict(reloaded.state_dict())
    assert original.keys() == restored.keys()
    for key in original:
        assert torch.allclose(original[key], restored[key]), f"state_dict mismatch at {key}"

    batch = make_batch(image_size=32, text_len=8)
    with torch.no_grad():
        a, b = model(batch), reloaded(batch)
    assert torch.allclose(a.pooled, b.pooled, atol=1e-6)
    assert a.text_tokens.shape == b.text_tokens.shape
    assert a.image_tokens.shape == b.image_tokens.shape


def test_config_json_is_the_model_config(tmp_path):
    model = RenaissanceModel(_tiny_config())
    model.save_pretrained(tmp_path)
    payload = json.loads((tmp_path / "config.json").read_text())
    # however the mixin nests it, the ModelConfig dict must be recoverable
    inner = payload.get("config", payload)
    assert ModelConfig.from_dict(inner) == model.config
