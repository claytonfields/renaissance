"""
Trainer smoke test: verifies that RenaissanceTrainer runs a couple of
steps on synthetic data without error and that the model still produces
a finite loss afterwards.

Reuses the shared two-tower pretrain fixtures from conftest so this
stays in sync with the rest of the modeling tests.
"""

import pytest
import torch

from renaissance.modeling import RenaissanceModel
from renaissance.trainer import RenaissanceTrainer

from tests.conftest import TWO_TOWER_IMG, _make_pretrain_batch


N_STEPS = 2


class _SyntheticLoader:
    """Iterable that yields the same batch a fixed number of times."""

    def __init__(self, batch, n=N_STEPS):
        self.n = n
        self._batch = batch

    def __iter__(self):
        for _ in range(self.n):
            yield self._batch

    def __len__(self):
        return self.n


@pytest.fixture(scope="class")
def trainer_config(two_tower_pretrain_config, tmp_path_factory):
    log_dir = tmp_path_factory.mktemp("trainer_smoke")
    return {
        **two_tower_pretrain_config,
        "max_steps": N_STEPS,
        "max_epoch": 1,
        "warmup_steps": 1,
        "log_dir": str(log_dir),
    }


class TestRenaissanceTrainer:

    @pytest.fixture(scope="class")
    def model(self, trainer_config):
        return RenaissanceModel(trainer_config)

    @pytest.fixture(scope="class")
    def batch(self):
        return _make_pretrain_batch(TWO_TOWER_IMG)

    @pytest.fixture(scope="class")
    def trainer(self, model, trainer_config, batch):
        loader = _SyntheticLoader(batch, n=N_STEPS)
        return RenaissanceTrainer(model, trainer_config, train_dataloader=loader)

    def test_trainer_instantiation(self, trainer):
        assert trainer.global_step == 0
        assert hasattr(trainer, "accelerator")
        assert hasattr(trainer, "optimizer")
        assert hasattr(trainer, "scheduler")

    def test_two_step_fit(self, trainer):
        """Run the training steps and verify global_step advances."""
        trainer.fit()
        assert trainer.global_step >= N_STEPS

    def test_loss_is_finite_after_training(self, trainer, batch):
        """Model should still produce a finite MLM loss after training."""
        model = trainer.accelerator.unwrap_model(trainer.model)
        model.eval()
        model.set_active_tasks()
        device = trainer.accelerator.device
        batch = RenaissanceTrainer._to_device(batch, device)
        with torch.no_grad():
            ret = model(batch)
        assert ret["mlm_loss"].isfinite().all()
