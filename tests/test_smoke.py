"""
Smoke tests — fast-dev-run level checks that verify each architecture and
objective function runs without error and produces finite losses.

These tests use synthetic batches (no real datasets required) and randomly
initialised encoder weights (no model downloads required beyond HF configs).
"""

import pytest
import torch

from renaissance.modules import RenaissanceTransformer
from renaissance.modules.objectives import (
    compute_mlm,
    compute_itm,
    compute_snli,
    compute_nlvr2,
    compute_ref,
)

from tests.conftest import BS, MAX_BB


def _finite(tensor):
    """Return True if all elements are finite (no NaN / Inf)."""
    return tensor.isfinite().all().item()


# ---------------------------------------------------------------------------
# One-Tower: pretrain (MLM + ITM)
# ---------------------------------------------------------------------------

class TestOneTowerPretrain:
    """Shared model instance keeps test runtime low — model creation is the
    expensive step (AutoConfig.from_pretrained downloads ~few KB)."""

    @pytest.fixture(scope="class")
    def model(self, one_tower_pretrain_config):
        return RenaissanceTransformer(one_tower_pretrain_config)

    def test_instantiation(self, model):
        assert model.model_type == "one-tower"
        assert hasattr(model, "mlm_score")
        assert hasattr(model, "itm_score")

    def test_infer_shape_and_values(self, model, one_tower_batch):
        model.eval()
        with torch.no_grad():
            out = model.infer(one_tower_batch)
        assert "cls_feats" in out
        assert "text_feats" in out
        assert "image_feats" in out
        assert out["cls_feats"].shape[0] == BS
        assert _finite(out["cls_feats"])

    def test_mlm_loss(self, model, one_tower_batch):
        model.train()
        ret = compute_mlm(model, one_tower_batch)
        assert _finite(ret["mlm_loss"])

    def test_itm_loss(self, model, one_tower_batch):
        model.train()
        ret = compute_itm(model, one_tower_batch)
        assert _finite(ret["itm_loss"])

    def test_combined_forward(self, model, one_tower_batch):
        """End-to-end forward through both active objectives."""
        model.train()
        model.current_tasks = ["mlm", "itm"]
        ret = model(one_tower_batch)
        total_loss = sum(v for k, v in ret.items() if "loss" in k)
        assert _finite(total_loss)


# ---------------------------------------------------------------------------
# Two-Tower: pretrain (MLM + ITM)
# ---------------------------------------------------------------------------

class TestTwoTowerPretrain:

    @pytest.fixture(scope="class")
    def model(self, two_tower_pretrain_config):
        return RenaissanceTransformer(two_tower_pretrain_config)

    def test_instantiation(self, model):
        assert model.model_type == "two-tower"
        assert hasattr(model, "mlm_score")
        assert hasattr(model, "itm_score")

    def test_infer_shape_and_values(self, model, two_tower_batch):
        model.eval()
        with torch.no_grad():
            out = model.infer(two_tower_batch)
        assert "cls_feats" in out
        assert out["cls_feats"].shape[0] == BS
        assert _finite(out["cls_feats"])

    def test_mlm_loss(self, model, two_tower_batch):
        model.train()
        ret = compute_mlm(model, two_tower_batch)
        assert _finite(ret["mlm_loss"])

    def test_itm_loss(self, model, two_tower_batch):
        model.train()
        ret = compute_itm(model, two_tower_batch)
        assert _finite(ret["itm_loss"])

    def test_combined_forward(self, model, two_tower_batch):
        model.train()
        model.current_tasks = ["mlm", "itm"]
        ret = model(two_tower_batch)
        total_loss = sum(v for k, v in ret.items() if "loss" in k)
        assert _finite(total_loss)


# ---------------------------------------------------------------------------
# Two-Tower: downstream fine-tuning tasks
# ---------------------------------------------------------------------------

class TestFinetuningSNLI:

    @pytest.fixture(scope="class")
    def model(self, two_tower_snli_config):
        return RenaissanceTransformer(two_tower_snli_config)

    def test_instantiation(self, model):
        assert hasattr(model, "snli_classifier")

    def test_snli_loss(self, model, snli_batch):
        model.train()
        ret = compute_snli(model, snli_batch)
        assert _finite(ret["snli_loss"])
        assert ret["snli_logits"].shape == (BS, 3)  # 3-class entailment

    def test_snli_forward(self, model, snli_batch):
        model.train()
        model.current_tasks = ["snli"]
        ret = model(snli_batch)
        assert _finite(ret["snli_loss"])


class TestFinetuningNLVR2:

    @pytest.fixture(scope="class")
    def model(self, two_tower_nlvr2_config):
        return RenaissanceTransformer(two_tower_nlvr2_config)

    def test_instantiation(self, model):
        assert hasattr(model, "nlvr2_classifier")

    def test_nlvr2_loss(self, model, nlvr2_batch):
        model.train()
        ret = compute_nlvr2(model, nlvr2_batch)
        assert _finite(ret["nlvr2_loss"])
        assert ret["nlvr2_logits"].shape == (BS, 2)  # binary


# ---------------------------------------------------------------------------
# Two-Tower: reference resolution
# ---------------------------------------------------------------------------

class TestRefResolution:

    @pytest.fixture(scope="class")
    def model(self, two_tower_ref_config):
        return RenaissanceTransformer(two_tower_ref_config)

    def test_instantiation(self, model):
        assert hasattr(model, "ref_classifier")

    def test_ref_loss(self, model, ref_batch):
        model.train()
        ret = compute_ref(model, ref_batch)
        assert _finite(ret["ref_loss"])
        # Logits should be (BS, MAX_BB) — one score per candidate region
        assert ret["ref_logits"].shape == (BS, MAX_BB)

    def test_ref_forward(self, model, ref_batch):
        model.train()
        model.current_tasks = ["ref"]
        ret = model(ref_batch)
        assert _finite(ret["ref_loss"])
