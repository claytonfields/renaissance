"""
Smoke tests — fast-dev-run checks that each architecture + objective runs
and produces finite losses, via `RenaissanceModel.forward`.

Synthetic batches, randomly-initialised encoders (no downloads beyond HF
configs). Overlaps intentionally with test_model.py/test_tasks.py but
keeps the one-tower coverage and the combined multi-task forward.
"""

import pytest
import torch

from renaissance.modeling import RenaissanceModel

from tests.conftest import BS, MAX_BB


def _finite(tensor):
    return torch.as_tensor(tensor).isfinite().all().item()


def _run(model, tasks, batch):
    model.train()
    model.current_tasks = list(tasks)
    return model(batch)


# ---------------------------------------------------------------------------
# One-Tower: pretrain (MLM + ITM)
# ---------------------------------------------------------------------------

class TestOneTowerPretrain:

    @pytest.fixture(scope="class")
    def model(self, one_tower_pretrain_config):
        return RenaissanceModel(one_tower_pretrain_config)

    def test_instantiation(self, model):
        assert model.config["model_type"] == "one-tower"
        assert "mlm" in model.heads
        assert "itm" in model.heads

    def test_infer_shape_and_values(self, model, one_tower_batch):
        model.eval()
        with torch.no_grad():
            out = model.infer(one_tower_batch)
        assert out["cls_feats"].shape[0] == BS
        assert _finite(out["cls_feats"])

    def test_mlm_loss(self, model, one_tower_batch):
        ret = _run(model, ["mlm"], one_tower_batch)
        assert _finite(ret["mlm_loss"])

    def test_itm_loss(self, model, one_tower_batch):
        ret = _run(model, ["itm"], one_tower_batch)
        assert _finite(ret["itm_loss"])

    def test_combined_forward(self, model, one_tower_batch):
        ret = _run(model, ["mlm", "itm"], one_tower_batch)
        total_loss = sum(v for k, v in ret.items() if k.endswith("_loss"))
        assert _finite(total_loss)


# ---------------------------------------------------------------------------
# Two-Tower: pretrain (MLM + ITM)
# ---------------------------------------------------------------------------

class TestTwoTowerPretrain:

    @pytest.fixture(scope="class")
    def model(self, two_tower_pretrain_config):
        return RenaissanceModel(two_tower_pretrain_config)

    def test_instantiation(self, model):
        assert model.config["model_type"] == "two-tower"
        assert "mlm" in model.heads
        assert "itm" in model.heads

    def test_infer_shape_and_values(self, model, two_tower_batch):
        model.eval()
        with torch.no_grad():
            out = model.infer(two_tower_batch)
        assert out["cls_feats"].shape[0] == BS
        assert _finite(out["cls_feats"])

    def test_mlm_loss(self, model, two_tower_batch):
        ret = _run(model, ["mlm"], two_tower_batch)
        assert _finite(ret["mlm_loss"])

    def test_itm_loss(self, model, two_tower_batch):
        ret = _run(model, ["itm"], two_tower_batch)
        assert _finite(ret["itm_loss"])

    def test_combined_forward(self, model, two_tower_batch):
        ret = _run(model, ["mlm", "itm"], two_tower_batch)
        total_loss = sum(v for k, v in ret.items() if k.endswith("_loss"))
        assert _finite(total_loss)


# ---------------------------------------------------------------------------
# Two-Tower: downstream fine-tuning tasks
# ---------------------------------------------------------------------------

class TestFinetuningSNLI:

    @pytest.fixture(scope="class")
    def model(self, two_tower_snli_config):
        return RenaissanceModel(two_tower_snli_config)

    def test_instantiation(self, model):
        assert "snli" in model.heads

    def test_snli_loss(self, model, snli_batch):
        ret = _run(model, ["snli"], snli_batch)
        assert _finite(ret["snli_loss"])
        assert ret["snli_logits"].shape == (BS, 3)


class TestFinetuningNLVR2:

    @pytest.fixture(scope="class")
    def model(self, two_tower_nlvr2_config):
        return RenaissanceModel(two_tower_nlvr2_config)

    def test_instantiation(self, model):
        assert "nlvr2" in model.heads

    def test_nlvr2_loss(self, model, nlvr2_batch):
        ret = _run(model, ["nlvr2"], nlvr2_batch)
        assert _finite(ret["nlvr2_loss"])
        assert ret["nlvr2_logits"].shape == (BS, 2)


class TestRefResolution:

    @pytest.fixture(scope="class")
    def model(self, two_tower_ref_config):
        return RenaissanceModel(two_tower_ref_config)

    def test_instantiation(self, model):
        assert "ref" in model.heads

    def test_ref_loss(self, model, ref_batch):
        ret = _run(model, ["ref"], ref_batch)
        assert _finite(ret["ref_loss"])
        assert ret["ref_logits"].shape == (BS, MAX_BB)
