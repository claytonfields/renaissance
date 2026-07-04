"""
Phase 1 — unit tests for `renaissance.modeling.backbones`.

These exercise the new backbone layer directly (it isn't wired into a
model yet). The cross-rewrite behavioral net is `test_contract.py`; this
file just pins the new protocol's own shape/contract.
"""

import pytest
import torch
from transformers import AutoConfig, AutoModel

from renaissance.modeling.backbones import (
    Backbone,
    EncoderOutput,
    OneTowerBackbone,
    TwoTowerBackbone,
    build_backbone,
    hf_model_hidden_size,
    load_hf_encoder,
)

from tests.conftest import BS

TEXT_ENC = "google/electra-small-discriminator"


def _finite(t):
    return torch.as_tensor(t).isfinite().all().item()


def _expected_pooled_dim(config):
    if config["model_type"] == "two-tower":
        return 2 * config["cross_layer_hidden_size"]
    if config["pooler_type"] == "single":
        return config["hidden_size"]
    return 2 * config["hidden_size"]


# ---------------------------------------------------------------------------
# hf_loader
# ---------------------------------------------------------------------------

class TestHFLoader:

    def test_hidden_size_from_plain_config(self):
        model = AutoModel.from_config(AutoConfig.from_pretrained(TEXT_ENC))
        # electra-small discriminator hidden size is 256.
        assert hf_model_hidden_size(model) == 256

    def test_random_init_with_overrides(self):
        model, hs = load_hf_encoder(
            TEXT_ENC, random_init=True, overrides={"hidden_size": 128, "num_hidden_layers": 2}
        )
        assert hs == 128
        assert model.config.num_hidden_layers == 2

    def test_overrides_rejected_for_pretrained(self):
        with pytest.raises(ValueError, match="random_init=True"):
            load_hf_encoder(TEXT_ENC, random_init=False, overrides={"hidden_size": 128})

    def test_freeze_disables_grad(self):
        model, _ = load_hf_encoder(TEXT_ENC, random_init=True, freeze=True)
        assert all(not p.requires_grad for p in model.parameters())


# ---------------------------------------------------------------------------
# build_backbone factory
# ---------------------------------------------------------------------------

class TestBuildBackbone:

    def test_two_tower_type(self, two_tower_pretrain_config):
        bb = build_backbone(two_tower_pretrain_config)
        assert isinstance(bb, (TwoTowerBackbone, Backbone))

    def test_one_tower_type(self, one_tower_pretrain_config):
        bb = build_backbone(one_tower_pretrain_config)
        assert isinstance(bb, (OneTowerBackbone, Backbone))

    def test_unknown_type_raises(self, two_tower_pretrain_config):
        cfg = dict(two_tower_pretrain_config)
        cfg["model_type"] = "three-tower"
        with pytest.raises(ValueError, match="Unknown model_type"):
            build_backbone(cfg)


class TestGradientCheckpointing:
    """Piece #2 of the Step 9 thin-perf slice: verify the
    ``gradient_checkpointing`` config flag reaches the HF encoders and that
    the default (off) is unchanged."""

    def test_two_tower_off_by_default(self, two_tower_pretrain_config):
        bb = build_backbone(two_tower_pretrain_config)
        assert not bb.encoder.image_encoder.is_gradient_checkpointing
        assert not bb.encoder.text_transformer.is_gradient_checkpointing

    def test_two_tower_on_when_flagged(self, two_tower_pretrain_config):
        cfg = {**two_tower_pretrain_config, "gradient_checkpointing": True}
        bb = build_backbone(cfg)
        assert bb.encoder.image_encoder.is_gradient_checkpointing
        assert bb.encoder.text_transformer.is_gradient_checkpointing

    def test_one_tower_off_by_default(self, one_tower_pretrain_config):
        bb = build_backbone(one_tower_pretrain_config)
        # OneTowerEncoder keeps only the inner Transformer stack; HF's
        # `gradient_checkpointing_enable` sets the flag on that stack directly.
        assert not bb.encoder.encoder.gradient_checkpointing

    def test_one_tower_on_when_flagged(self, one_tower_pretrain_config):
        cfg = {**one_tower_pretrain_config, "gradient_checkpointing": True}
        bb = build_backbone(cfg)
        assert bb.encoder.encoder.gradient_checkpointing


# ---------------------------------------------------------------------------
# EncoderOutput container
# ---------------------------------------------------------------------------

class TestEncoderOutput:

    def test_legacy_aliases(self):
        p = torch.zeros(2, 4)
        t = torch.zeros(2, 3, 5)
        i = torch.zeros(2, 7, 5)
        out = EncoderOutput(pooled=p, text_tokens=t, image_tokens=i)
        assert out["cls_feats"] is p
        assert out["text_feats"] is t
        assert out["image_feats"] is i
        assert "cls_feats" in out
        assert "text_masks" not in out  # None → absent
        assert out.get("text_masks") is None


# ---------------------------------------------------------------------------
# Forward contract (wraps legacy encoders for now)
# ---------------------------------------------------------------------------

class TestTwoTowerForward:

    @pytest.fixture(scope="class")
    def backbone(self, two_tower_pretrain_config):
        return build_backbone(two_tower_pretrain_config)

    def test_output_shapes(self, backbone, two_tower_pretrain_config, two_tower_batch):
        backbone.eval()
        with torch.no_grad():
            out = backbone(two_tower_batch)
        assert isinstance(out, EncoderOutput)
        assert out.pooled.shape == (BS, _expected_pooled_dim(two_tower_pretrain_config))
        assert out.pooled.shape[-1] == backbone.pooled_dim
        assert out.text_tokens.ndim == 3
        assert out.image_tokens.ndim == 3
        assert _finite(out.pooled)

    def test_stream_hidden_sizes(self, backbone):
        assert isinstance(backbone.text_hidden_size, int)
        assert isinstance(backbone.image_hidden_size, int)

    def test_mask_text_propagates_labels(self, backbone, two_tower_batch):
        backbone.train()
        out = backbone(two_tower_batch, mask_text=True)
        assert out.text_labels is not None
        assert out.text_labels.shape[0] == BS


class TestOneTowerForward:

    @pytest.fixture(scope="class")
    def backbone(self, one_tower_pretrain_config):
        return build_backbone(one_tower_pretrain_config)

    def test_output_shapes(self, backbone, one_tower_pretrain_config, one_tower_batch):
        backbone.eval()
        with torch.no_grad():
            out = backbone(one_tower_batch)
        assert out.pooled.shape == (BS, _expected_pooled_dim(one_tower_pretrain_config))
        assert out.pooled.shape[-1] == backbone.pooled_dim
        assert _finite(out.pooled)

    def test_encode_text_only(self, backbone, mrpc_batch):
        # Text-only path consumes the GLUE-style {input_ids, attention_mask}
        # schema, not the pretrain {text_ids} one.
        backbone.eval()
        with torch.no_grad():
            hidden = backbone.encode_text_only(mrpc_batch)
        assert torch.is_tensor(hidden)
        assert hidden.shape[0] == BS
