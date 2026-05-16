"""
RenaissanceModel (Phase 4).

The whole point of the rewrite: `__init__` and `forward` are now tiny
registry loops instead of the legacy 200-line copy-paste `__init__` and
70-line if-ladder `forward`.

  __init__:  build backbone → for each active task, build its head
  forward:   for each task in current_tasks, run it → collect outputs

Adding a task touches neither method — only `TASK_REGISTRY`.

Phase 4 still reads the active set from `config["loss_names"]` (the
existing schema); Phase 5 swaps in a `tasks` list with a `loss_names`
shim. Metric *objects* (torchmetrics, train/dev/test routing) are still
deferred — `forward` logs the per-task loss to `_log_buffer` for the
trainer; Phase 6 wires the richer metrics. Legacy `renaissance/modules/`
is untouched; the test/`run.py` cutover is Phase 7.
"""

import os

import torch
import torch.nn as nn

from .backbones import build_backbone
from .tasks import TASK_REGISTRY


def _active_task_names(config):
    """Tasks with a head to build, filtered to registered Tasks.

    Prefers the normalized ``tasks`` list (Phase 5 schema); falls back to
    deriving from ``loss_names`` so configs built directly as flat dicts
    (tests, older callers) still work without running `normalize_tasks`.
    Stub GLUE names / irtr that have no registered Task are dropped (the
    rewrite removes those dead branches)."""
    tasks = config.get("tasks")
    if not tasks:
        tasks = [
            name
            for name, weight in config.get("loss_names", {}).items()
            if weight and weight > 0
        ]
    return [name for name in tasks if name in TASK_REGISTRY]


class RenaissanceModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = build_backbone(config)

        self.heads = nn.ModuleDict()
        self._metric_names = {}
        for name in _active_task_names(config):
            task = TASK_REGISTRY[name]
            head = task.build_head(self.backbone, config)
            if head is not None:
                self.heads[name] = head
            self._metric_names[name] = task.metric_names()

        self.current_tasks = []
        self._log_buffer: dict = {}

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def infer(self, batch, **kwargs):
        """Encoder-only forward. Returns an `EncoderOutput` (supports both
        `.pooled` and the legacy `["cls_feats"]` access)."""
        return self.backbone(batch, **kwargs)

    def forward(self, batch):
        if not self.current_tasks:
            out = self.infer(batch)
            return {
                "cls_feats": out.pooled,
                "text_feats": out.text_tokens,
                "image_feats": out.image_tokens,
            }

        ret = {}
        for name in self.current_tasks:
            task = TASK_REGISTRY[name]
            if name not in self.heads and task.build_head(self.backbone, self.config) is not None:
                raise RuntimeError(
                    f"Task {name!r} is active but its head was never built — "
                    f"it isn't in config['loss_names']."
                )
            out = task.forward(self, batch)
            ret[f"{name}_loss"] = out.loss
            if out.logits is not None:
                ret[f"{name}_logits"] = out.logits
            if out.targets is not None:
                ret[f"{name}_targets"] = out.targets
            ret.update({f"{name}_{k}": v for k, v in out.extras.items()})
            self.log(f"{name}/loss", out.loss)
        return ret

    # ------------------------------------------------------------------ #
    # Trainer-facing helpers (kept from the legacy interface)
    # ------------------------------------------------------------------ #

    @property
    def device(self):
        return next(self.parameters()).device

    def log(self, name, value, **kwargs):
        if torch.is_tensor(value):
            value = value.item()
        self._log_buffer[name] = value

    # ------------------------------------------------------------------ #
    # Persistence (same on-disk format as the legacy model)
    # ------------------------------------------------------------------ #

    def save_pretrained(self, path: str) -> None:
        from safetensors.torch import save_file

        from renaissance.hub import RenaissanceHubConfig

        os.makedirs(path, exist_ok=True)
        RenaissanceHubConfig.from_flat_config(self.config).save_pretrained(path)
        state_dict = {k: v.contiguous().cpu() for k, v in self.state_dict().items()}
        save_file(state_dict, os.path.join(path, "model.safetensors"))

    @classmethod
    def from_pretrained(cls, path_or_repo_id: str) -> "RenaissanceModel":
        from safetensors.torch import load_file

        from renaissance.hub import RenaissanceHubConfig

        hub_cfg = RenaissanceHubConfig.from_pretrained(path_or_repo_id)
        model = cls(hub_cfg.to_flat_config())

        if os.path.isdir(path_or_repo_id):
            weights_path = os.path.join(path_or_repo_id, "model.safetensors")
        else:
            from huggingface_hub import hf_hub_download

            weights_path = hf_hub_download(path_or_repo_id, "model.safetensors")

        model.load_state_dict(load_file(weights_path), strict=False)
        return model
