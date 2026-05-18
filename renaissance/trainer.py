import os
import warnings

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from .modeling.optim import set_schedule


def _make_summary_writer(log_dir):
    """Construct a TensorBoard SummaryWriter, degrading to None if the
    tensorboard backend is unavailable or broken.

    `torch.utils.tensorboard` is imported lazily here (not at module
    load) so that importing the trainer never drags in tensorboard /
    tensorflow. Logging is best-effort: a missing or broken backend must
    not crash training — every `self.writer` use is already None-guarded.
    """
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(log_dir=log_dir)
    except Exception as e:  # ImportError, or tensorboard's TF fallback failing
        warnings.warn(
            f"TensorBoard logging disabled — could not initialize SummaryWriter: {e}",
            RuntimeWarning,
        )
        return None


class RenaissanceTrainer:
    """Accelerate-based training loop replacing PyTorch Lightning."""

    def __init__(self, model, config, train_dataloader, val_dataloader=None):
        self.model = model
        self.config = config

        num_gpus = config.get("num_gpus", 1)
        if isinstance(num_gpus, list):
            num_gpus = len(num_gpus)
        num_nodes = config.get("num_nodes", 1)
        grad_accum = max(
            config["batch_size"] // (config["per_gpu_batchsize"] * num_gpus * num_nodes),
            1,
        )

        precision = config.get("precision", 32)
        mixed_precision = "fp16" if precision == 16 else ("bf16" if precision == "bf16" else "no")

        # DDP: allow unused parameters (some heads are inactive depending on active tasks)
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=grad_accum,
            kwargs_handlers=[ddp_kwargs],
        )

        self.max_steps: int = config.get("max_steps") or 10 ** 9
        self.max_epochs: int = config.get("max_epoch", 1)

        # Build schedule before prepare() so optimizer sees unwrapped params
        optimizer, scheduler = set_schedule(model, config, self.max_steps)
        self.optimizer = optimizer
        self.scheduler = scheduler

        if val_dataloader is not None:
            (
                self.model,
                self.optimizer,
                self.train_dataloader,
                self.val_dataloader,
            ) = self.accelerator.prepare(model, optimizer, train_dataloader, val_dataloader)
        else:
            self.model, self.optimizer, self.train_dataloader = self.accelerator.prepare(
                model, optimizer, train_dataloader
            )
            self.val_dataloader = None

        self.global_step = 0
        self.current_epoch = 0

        log_dir = config.get("log_dir", "result")
        os.makedirs(log_dir, exist_ok=True)
        self.writer = _make_summary_writer(log_dir) if self.accelerator.is_main_process else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self):
        for epoch in range(self.max_epochs):
            self.current_epoch = epoch
            self._train_epoch()
            if self.val_dataloader is not None:
                self._val_epoch()
            if self.global_step >= self.max_steps:
                break
        if self.writer is not None:
            self.writer.close()

    def test(self):
        self._eval_epoch(phase="test", dataloader=self.val_dataloader)

    def save_checkpoint(self, path: str):
        if self.accelerator.is_main_process:
            unwrapped = self.accelerator.unwrap_model(self.model)
            unwrapped.save_pretrained(path)
        self.accelerator.save_state(os.path.join(path, "training_state"))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _to_device(batch, device):
        """Recursively move tensors in a batch dict/list to device."""
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        if isinstance(batch, dict):
            return {k: RenaissanceTrainer._to_device(v, device) for k, v in batch.items()}
        if isinstance(batch, (list, tuple)):
            moved = [RenaissanceTrainer._to_device(v, device) for v in batch]
            return type(batch)(moved)
        return batch

    def _train_epoch(self):
        self.model.train()
        self.model.set_active_tasks()
        device = self.accelerator.device

        for batch in self.train_dataloader:
            batch = self._to_device(batch, device)
            self.model._log_buffer = {}
            with self.accelerator.accumulate(self.model):
                output = self.model(batch)
                # RenaissanceModel.forward emits per-task losses as
                # exactly "<task>_loss"; match that precisely rather than
                # any key containing the substring "loss" (task `extras`
                # are emitted as "<task>_<key>" and could collide).
                total_loss = sum(v for k, v in output.items() if k.endswith("_loss"))
                self.accelerator.backward(total_loss)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            if self.accelerator.is_main_process and self.writer is not None:
                for k, v in self.model._log_buffer.items():
                    self.writer.add_scalar(k, v, self.global_step)
                self.writer.add_scalar("train/loss_step", total_loss.item(), self.global_step)

            self.global_step += 1
            if self.global_step >= self.max_steps:
                break

        self._log_metrics(self.model.epoch_metrics("train"))

    def _val_epoch(self):
        self._eval_epoch(phase="val", dataloader=self.val_dataloader)

    def _eval_epoch(self, phase: str, dataloader):
        self.model.eval()
        self.model.set_active_tasks()
        device = self.accelerator.device
        with torch.no_grad():
            for batch in dataloader:
                batch = self._to_device(batch, device)
                self.model._log_buffer = {}
                self.model(batch)

        self._log_metrics(self.model.epoch_metrics(phase))

    def _log_metrics(self, metrics: dict):
        if self.accelerator.is_main_process and self.writer is not None:
            for k, v in metrics.items():
                self.writer.add_scalar(k, v, self.current_epoch)
