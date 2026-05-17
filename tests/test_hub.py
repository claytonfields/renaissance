"""
Round-trip tests for Step 5 — Hub integration + safetensors.

save_pretrained → from_pretrained must reproduce identical parameters
and identical forward outputs.
"""

import os
import tempfile

import pytest
import torch

from renaissance.hub import RenaissanceHubConfig
from renaissance.modeling import RenaissanceModel


# ---------------------------------------------------------------------------
# Config serialisation
# ---------------------------------------------------------------------------

class TestRenaissanceHubConfig:
    def test_from_flat_preserves_model_fields(self, two_tower_pretrain_config):
        hub_cfg = RenaissanceHubConfig.from_flat_config(two_tower_pretrain_config)
        assert hub_cfg.arch == "two-tower"
        assert hub_cfg.image_encoder == two_tower_pretrain_config["image_encoder"]
        assert hub_cfg.cross_layer_hidden_size == two_tower_pretrain_config["cross_layer_hidden_size"]
        assert hub_cfg.loss_names == two_tower_pretrain_config["loss_names"]

    def test_to_flat_restores_model_type(self, two_tower_pretrain_config):
        hub_cfg = RenaissanceHubConfig.from_flat_config(two_tower_pretrain_config)
        flat = hub_cfg.to_flat_config()
        assert flat["model_type"] == "two-tower"
        assert flat["load_path"] == ""
        assert flat["test_only"] is False

    def test_json_round_trip(self, two_tower_pretrain_config):
        """save_pretrained (config only) → from_pretrained preserves all fields."""
        hub_cfg = RenaissanceHubConfig.from_flat_config(two_tower_pretrain_config)
        with tempfile.TemporaryDirectory() as tmp:
            hub_cfg.save_pretrained(tmp)
            assert os.path.exists(os.path.join(tmp, "config.json"))
            loaded = RenaissanceHubConfig.from_pretrained(tmp)

        assert loaded.arch == hub_cfg.arch
        assert loaded.cross_layer_hidden_size == hub_cfg.cross_layer_hidden_size
        assert loaded.num_cross_layers == hub_cfg.num_cross_layers
        assert loaded.loss_names == hub_cfg.loss_names
        assert loaded.vqav2_label_size == hub_cfg.vqav2_label_size

    def test_one_tower_config_round_trip(self, one_tower_pretrain_config):
        hub_cfg = RenaissanceHubConfig.from_flat_config(one_tower_pretrain_config)
        assert hub_cfg.arch == "one-tower"
        flat = hub_cfg.to_flat_config()
        assert flat["model_type"] == "one-tower"
        assert flat["encoder"] == one_tower_pretrain_config["encoder"]


# ---------------------------------------------------------------------------
# Checkpoint files
# ---------------------------------------------------------------------------

class TestSavePretrained:
    def test_creates_config_json(self, two_tower_pretrain_config):
        model = RenaissanceModel(two_tower_pretrain_config)
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            assert os.path.exists(os.path.join(tmp, "config.json"))

    def test_creates_safetensors(self, two_tower_pretrain_config):
        model = RenaissanceModel(two_tower_pretrain_config)
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            assert os.path.exists(os.path.join(tmp, "model.safetensors"))

    def test_safetensors_contains_all_params(self, two_tower_pretrain_config):
        from safetensors.torch import load_file

        model = RenaissanceModel(two_tower_pretrain_config)
        model.eval()
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            st = load_file(os.path.join(tmp, "model.safetensors"))

        orig_keys = set(model.state_dict().keys())
        assert st.keys() == orig_keys


# ---------------------------------------------------------------------------
# Round-trip: param equality + forward equality
# ---------------------------------------------------------------------------

class TestFromPretrained:
    def test_param_equality(self, two_tower_pretrain_config):
        model = RenaissanceModel(two_tower_pretrain_config)
        model.eval()
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            loaded = RenaissanceModel.from_pretrained(tmp)
        loaded.eval()

        orig = dict(model.named_parameters())
        for name, param in loaded.named_parameters():
            assert name in orig, f"unexpected param after round-trip: {name}"
            assert torch.allclose(param, orig[name]), f"param mismatch: {name}"

    def test_forward_output_equality(self, two_tower_pretrain_config, two_tower_batch):
        model = RenaissanceModel(two_tower_pretrain_config)
        model.eval()
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            loaded = RenaissanceModel.from_pretrained(tmp)
        loaded.eval()

        with torch.no_grad():
            out_orig = model.infer(two_tower_batch)
            out_loaded = loaded.infer(two_tower_batch)

        for key in ("cls_feats", "text_feats", "image_feats"):
            assert torch.allclose(out_orig[key], out_loaded[key], atol=1e-6), \
                f"output mismatch at '{key}'"
