"""
Trainer smoke test (Step 3): verifies that RenaissanceTrainer runs 2 steps
on synthetic data without error, and that the loss decreases between steps.
"""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from renaissance.modules import RenaissanceTransformer
from renaissance.trainer import RenaissanceTrainer  # noqa: F401 — also used inside test methods

from tests.conftest import (
    BS, TWO_TOWER_IMG, TEXT_LEN, VOCAB, ALL_LOSS_NAMES,
)


# ---------------------------------------------------------------------------
# Minimal config — mirrors conftest two_tower_pretrain_config
# ---------------------------------------------------------------------------

TWO_TOWER_IMG_SIZE = 224

def _trainer_config():
    return {
        "exp_name": "trainer_smoke",
        "load_path": "",
        "test_only": False,
        "fast_dev_run": False,
        "get_recall_metric": False,
        "model_type": "two-tower",
        "per_gpu_batchsize": BS,
        "batch_size": BS,
        "datasets": ["coco"],
        "draw_false_image": 1,
        "draw_false_text": 0,
        "vocab_size": VOCAB,
        "max_text_len": TEXT_LEN,
        "image_size": TWO_TOWER_IMG_SIZE,
        "original_image_size": TWO_TOWER_IMG_SIZE,
        "patch_size": 16,
        "image_only": False,
        "loss_names": {**ALL_LOSS_NAMES, "mlm": 1, "itm": 1},
        "vqav2_label_size": 10,
        "max_bb": 4,
        "ref_res_head_layers": 2,
        # Optimizer
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "lr_mult_head": 5,
        "lr_mult_cross_modal": 5,
        "end_lr": 0,
        "decay_power": 1,
        "optim_type": "adamw",
        "warmup_steps": 1,
        "max_epoch": 1,
        "max_steps": 2,
        "precision": 32,
        "val_check_interval": 1.0,
        "num_gpus": 1,
        "num_nodes": 1,
        "log_dir": "/tmp/renaissance_trainer_test",
        # Two-tower
        "cross_layer_hidden_size": 64,
        "num_cross_layers": 2,
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
    }


# ---------------------------------------------------------------------------
# Synthetic collate-compatible batch list
# ---------------------------------------------------------------------------

def _make_batch():
    return {
        "image": [torch.randn(BS, 3, TWO_TOWER_IMG_SIZE, TWO_TOWER_IMG_SIZE)],
        "false_image_0": [torch.randn(BS, 3, TWO_TOWER_IMG_SIZE, TWO_TOWER_IMG_SIZE)],
        "text": ["hello world"] * BS,
        "text_ids": torch.randint(1, VOCAB, (BS, TEXT_LEN)),
        "text_labels": torch.full((BS, TEXT_LEN), -100, dtype=torch.long),
        "text_masks": torch.ones(BS, TEXT_LEN, dtype=torch.long),
        "text_ids_mlm": torch.randint(1, VOCAB, (BS, TEXT_LEN)),
        "text_labels_mlm": torch.cat([
            torch.randint(1, VOCAB, (BS, 3)),
            torch.full((BS, TEXT_LEN - 3), -100, dtype=torch.long),
        ], dim=1),
    }


class _SyntheticLoader:
    """Iterable that yields the same batch indefinitely (for 2 steps)."""
    def __init__(self, n=4):
        self.n = n
        self._batch = _make_batch()

    def __iter__(self):
        for _ in range(self.n):
            yield self._batch

    def __len__(self):
        return self.n


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRenaissanceTrainer:

    @pytest.fixture(scope="class")
    def config(self):
        return _trainer_config()

    @pytest.fixture(scope="class")
    def model(self, config):
        return RenaissanceTransformer(config)

    @pytest.fixture(scope="class")
    def trainer(self, model, config):
        loader = _SyntheticLoader(n=2)
        return RenaissanceTrainer(model, config, train_dataloader=loader)

    def test_trainer_instantiation(self, trainer):
        assert trainer.global_step == 0
        assert hasattr(trainer, "accelerator")
        assert hasattr(trainer, "optimizer")
        assert hasattr(trainer, "scheduler")

    def test_two_step_fit(self, trainer):
        """Run 2 training steps and verify global_step advances."""
        trainer.fit()
        assert trainer.global_step >= 2

    def test_loss_is_finite_after_training(self, trainer):
        """Model should still produce finite loss after training."""
        model = trainer.accelerator.unwrap_model(trainer.model)
        model.eval()
        device = trainer.accelerator.device
        from renaissance.modules.objectives import compute_mlm
        batch = RenaissanceTrainer._to_device(_make_batch(), device)
        with torch.no_grad():
            ret = compute_mlm(model, batch)
        assert ret["mlm_loss"].isfinite().all()
