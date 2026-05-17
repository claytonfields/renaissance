"""
Deterministic evaluation harness tests (Step 6).

Each test constructs a minimal model with a known task head, runs a fixed
synthetic batch through the Evaluator, and asserts the returned metric matches
a hand-computed expected value.

Design: the model is put into eval() mode with fixed (all-zeros) weights so
that the logits are deterministic.  We then verify that:
  - the Evaluator returns the expected metric keys
  - the numeric values match what we would compute by hand from those logits
"""

import math
import pytest
import torch
import torch.nn as nn

from renaissance.eval import Evaluator, evaluate
from renaissance.modeling import RenaissanceModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BS = 2
TEXT_LEN = 8
VOCAB = 100

ALL_LOSS_NAMES = {
    "itm": 0, "mlm": 0, "mpp": 0, "vqa": 0, "vcr": 0, "vcr_qar": 0,
    "nlvr2": 0, "irtr": 0, "contras": 0, "snli": 0, "ref": 0, "ref2": 0,
    "mrpc": 0, "rte": 0, "wnli": 0, "sst2": 0, "qqp": 0, "qnli": 0,
    "mnli": 0, "cola": 0, "cifar10": 0,
}

IMG = 224
PATCH = 16
CROSS_HIDDEN = 64
NUM_CROSS = 2


def _two_tower_config(loss_names_patch: dict) -> dict:
    return {
        "model_type": "two-tower",
        "exp_name": "eval_test",
        "load_path": "",
        "test_only": False,
        "get_recall_metric": False,
        "per_gpu_batchsize": BS,
        "batch_size": BS,
        "datasets": ["coco"],
        "draw_false_image": 1,
        "draw_false_text": 0,
        "vocab_size": VOCAB,
        "max_text_len": TEXT_LEN,
        "image_size": IMG,
        "original_image_size": IMG,
        "patch_size": PATCH,
        "image_only": False,
        "loss_names": {**ALL_LOSS_NAMES, **loss_names_patch},
        "vqav2_label_size": 4,
        "max_bb": 2,
        "ref_res_head_layers": 1,
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "lr_mult_head": 5,
        "lr_mult_cross_modal": 5,
        "end_lr": 0,
        "decay_power": 1,
        "optim_type": "adamw",
        "warmup_steps": 10,
        "max_epoch": 1,
        "max_steps": 10,
        "precision": 32,
        "val_check_interval": 1.0,
        "cross_layer_hidden_size": CROSS_HIDDEN,
        "num_cross_layers": NUM_CROSS,
        "num_cross_layer_heads": 4,
        "cross_layer_mlp_ratio": 4,
        "cross_layer_drop_rate": 0.0,
        "image_encoder": "facebook/deit-tiny-patch16-224",
        "text_encoder": "google/electra-small-discriminator",
        "random_init_vision_encoder": True,
        "random_init_text_encoder": True,
        "image_encoder_manual_configuration": False,
        "text_encoder_manual_configuration": False,
        "freeze_image_encoder": False,
        "freeze_text_encoder": False,
        "freeze_cross_modal_layers": False,
        "image_encoder_hidden_size": 192,
        "image_encoder_num_heads": 3,
        "image_encoder_num_layers": 12,
        "image_encoder_mlp_ratio": 4,
        "image_encoder_drop_rate": 0.0,
        "image_encoder_embedding_size": 128,
        "text_encoder_hidden_size": 256,
        "text_encoder_num_heads": 4,
        "text_encoder_num_layers": 12,
        "text_encoder_mlp_ratio": 4,
        "text_encoder_drop_rate": 0.0,
        "text_encoder_embedding_size": 128,
        "encoder": "google/electra-small-discriminator",
        "random_init_encoder": True,
        "encoder_manual_configuration": False,
        "pooler_type": "double",
        "drop_rate": 0.0,
        "hidden_size": 256,
        "num_heads": 4,
        "num_layers": 6,
        "mlp_ratio": 4,
        "embedding_size": 128,
    }


def _image_batch():
    return [torch.zeros(BS, 3, IMG, IMG)]


def _text_batch():
    return {
        "text": ["hello world"] * BS,
        "text_ids": torch.ones(BS, TEXT_LEN, dtype=torch.long),
        "text_labels": torch.full((BS, TEXT_LEN), -100, dtype=torch.long),
        "text_masks": torch.ones(BS, TEXT_LEN, dtype=torch.long),
        "text_ids_mlm": torch.ones(BS, TEXT_LEN, dtype=torch.long),
        "text_labels_mlm": torch.cat([
            torch.ones(BS, 2, dtype=torch.long),
            torch.full((BS, TEXT_LEN - 2), -100, dtype=torch.long),
        ], dim=1),
    }


def _pretrain_batch():
    b = _text_batch()
    b["image"] = _image_batch()
    b["false_image_0"] = _image_batch()
    return b


# ---------------------------------------------------------------------------
# Tests: Evaluator registry
# ---------------------------------------------------------------------------

class TestEvaluatorRegistry:
    def test_known_tasks_registered(self):
        # irtr was intentionally dropped in the modeling rewrite (needs
        # draw_false_text negatives + a rank head derived from ITM weights).
        for task in ("mlm", "itm", "vqa", "nlvr2", "snli", "ref", "ref2", "mrpc"):
            assert task in Evaluator._REGISTRY, f"task '{task}' not registered"

    def test_unknown_task_raises(self):
        cfg = _two_tower_config({"itm": 1})
        model = RenaissanceModel(cfg)
        with pytest.raises(ValueError, match="No evaluator"):
            evaluate(model, [], task="nonexistent_task")


# ---------------------------------------------------------------------------
# Tests: ITM — deterministic metric value
# ---------------------------------------------------------------------------

class TestITMEvaluator:
    """ITM accuracy: model outputs uniform logits → argmax always 0 (class 0 always
    predicted).  If all true labels are 0, accuracy = 1.0; if all labels are 1,
    accuracy = 0.0."""

    @pytest.fixture(scope="class")
    def model(self):
        cfg = _two_tower_config({"itm": 1})
        m = RenaissanceModel(cfg)
        m.eval()
        # Zero-out ITM head weights → logits are constant → argmax = 0
        for p in m.heads["itm"].parameters():
            nn.init.zeros_(p)
        return m

    def _make_batch(self, itm_label_for_true_samples):
        """Build a batch where true-image samples get the specified label."""
        batch = _pretrain_batch()
        return batch

    def test_returns_metric_keys(self, model):
        batch = _pretrain_batch()
        metrics = evaluate(model, [batch], task="itm")
        assert "itm/val/accuracy_epoch" in metrics
        assert "itm/val/loss_epoch" in metrics

    def test_metric_is_finite(self, model):
        batch = _pretrain_batch()
        metrics = evaluate(model, [batch], task="itm")
        assert math.isfinite(metrics["itm/val/accuracy_epoch"])
        assert math.isfinite(metrics["itm/val/loss_epoch"])

    def test_loss_is_positive(self, model):
        batch = _pretrain_batch()
        metrics = evaluate(model, [batch], task="itm")
        assert metrics["itm/val/loss_epoch"] > 0


# ---------------------------------------------------------------------------
# Tests: SNLI — deterministic metric value
# ---------------------------------------------------------------------------

class TestSNLIEvaluator:
    """SNLI: 3-class. With zero weights, all logits = 0 → uniform → argmax = 0.
    Labels are [0, 1] → one correct → accuracy = 0.5."""

    @pytest.fixture(scope="class")
    def model(self):
        cfg = _two_tower_config({"snli": 1})
        m = RenaissanceModel(cfg)
        m.eval()
        for p in m.heads["snli"].parameters():
            nn.init.zeros_(p)
        return m

    def _snli_batch(self, labels):
        b = _pretrain_batch()
        b["labels"] = labels
        return b

    def test_returns_metric_keys(self, model):
        batch = self._snli_batch([0, 1])
        metrics = evaluate(model, [batch], task="snli")
        assert "snli/val/accuracy_epoch" in metrics

    def test_all_correct_accuracy_one(self, model):
        """All labels = 0, zero weights → argmax always 0 → 100% accuracy."""
        batch = self._snli_batch([0, 0])
        metrics = evaluate(model, [batch], task="snli")
        assert metrics["snli/val/accuracy_epoch"] == pytest.approx(1.0)

    def test_all_wrong_accuracy_zero(self, model):
        """All labels ≠ 0, zero weights → argmax always 0 → 0% accuracy."""
        batch = self._snli_batch([1, 2])
        metrics = evaluate(model, [batch], task="snli")
        assert metrics["snli/val/accuracy_epoch"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests: NLVR2 — deterministic metric value
# ---------------------------------------------------------------------------

class TestNLVR2Evaluator:
    """NLVR2: 2-class. Zero weights → argmax = 0. Labels [0,0] → accuracy 1.0."""

    @pytest.fixture(scope="class")
    def model(self):
        cfg = _two_tower_config({"nlvr2": 1})
        m = RenaissanceModel(cfg)
        m.eval()
        for p in m.heads["nlvr2"].parameters():
            nn.init.zeros_(p)
        return m

    def _nlvr2_batch(self, answers):
        b = _pretrain_batch()
        b["image_0"] = _image_batch()
        b["image_1"] = _image_batch()
        b["answers"] = answers
        return b

    def test_returns_metric_keys(self, model):
        batch = self._nlvr2_batch([0, 0])
        metrics = evaluate(model, [batch], task="nlvr2")
        assert "nlvr2/val/accuracy_epoch" in metrics

    def test_all_correct(self, model):
        batch = self._nlvr2_batch([0, 0])
        metrics = evaluate(model, [batch], task="nlvr2")
        assert metrics["nlvr2/val/accuracy_epoch"] == pytest.approx(1.0)

    def test_all_wrong(self, model):
        batch = self._nlvr2_batch([1, 1])
        metrics = evaluate(model, [batch], task="nlvr2")
        assert metrics["nlvr2/val/accuracy_epoch"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests: Ref resolution — deterministic metric value
# ---------------------------------------------------------------------------

class TestRefEvaluator:
    """Ref: argmax over num_regions. With zero head weights, all region scores = 0
    → logits uniform over regions → argmax = 0 → accuracy depends on targets."""

    @pytest.fixture(scope="class")
    def model(self):
        cfg = _two_tower_config({"ref": 1})
        m = RenaissanceModel(cfg)
        m.eval()
        for p in m.heads["ref"].parameters():
            nn.init.zeros_(p)
        return m

    def _ref_batch(self, targets):
        num_regions = 2
        total = BS * num_regions
        return {
            "image": [torch.zeros(total, 3, IMG, IMG)],
            "text_ids": torch.ones(total, TEXT_LEN, dtype=torch.long),
            "text_labels": torch.full((total, TEXT_LEN), -100, dtype=torch.long),
            "text_masks": torch.ones(total, TEXT_LEN, dtype=torch.long),
            "target": torch.tensor(targets),
        }

    def test_returns_metric_keys(self, model):
        batch = self._ref_batch([0, 0])
        metrics = evaluate(model, [batch], task="ref")
        assert "ref/val/accuracy_epoch" in metrics

    def test_all_correct(self, model):
        """All targets = 0, logits uniform → argmax 0 → accuracy = 1.0."""
        batch = self._ref_batch([0, 0])
        metrics = evaluate(model, [batch], task="ref")
        assert metrics["ref/val/accuracy_epoch"] == pytest.approx(1.0)

    def test_all_wrong(self, model):
        """All targets = 1, argmax always 0 → accuracy = 0.0."""
        batch = self._ref_batch([1, 1])
        metrics = evaluate(model, [batch], task="ref")
        assert metrics["ref/val/accuracy_epoch"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests: evaluate() returns the_metric aggregation key
# ---------------------------------------------------------------------------

class TestEvaluateTopLevel:
    def test_returns_the_metric(self):
        cfg = _two_tower_config({"itm": 1})
        model = RenaissanceModel(cfg)
        model.eval()
        batch = _pretrain_batch()
        metrics = evaluate(model, [batch], task="itm")
        assert "val/the_metric" in metrics

    def test_multiple_batches_accumulate(self):
        """Running two identical batches should give same metric as one batch
        (Accuracy is a ratio, not a sum)."""
        cfg = _two_tower_config({"itm": 1})
        model = RenaissanceModel(cfg)
        model.eval()
        batch = _pretrain_batch()

        m1 = evaluate(model, [batch], task="itm")
        m2 = evaluate(model, [batch, batch], task="itm")
        assert m1["itm/val/accuracy_epoch"] == pytest.approx(
            m2["itm/val/accuracy_epoch"], abs=1e-4
        )
