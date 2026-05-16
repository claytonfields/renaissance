"""Masked language modeling."""

import torch.nn.functional as F

from ..heads import MlmHead, init_weights
from .base import Task, TaskOutput


class MlmTask(Task):
    name = "mlm"

    def build_head(self, backbone, config):
        head = MlmHead(backbone.token_dim, config["vocab_size"])
        head.apply(init_weights)
        return head

    def metric_names(self):
        return ("accuracy",)

    def forward(self, model, batch) -> TaskOutput:
        out = model.backbone(batch, mask_text=True, mask_image=False)
        logits = model.heads[self.name](out.text_tokens)
        labels = out.text_labels
        loss = F.cross_entropy(
            logits.view(-1, model.config["vocab_size"]),
            labels.view(-1),
            ignore_index=-100,
        )
        return TaskOutput(loss=loss, logits=logits, targets=labels)
