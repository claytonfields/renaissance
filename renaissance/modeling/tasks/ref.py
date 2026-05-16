"""Referring-expression resolution: pick the matching region.

`BS * num_regions` candidate crops are stacked along the batch dim; the
head scores each, logits are reshaped to `(BS, num_regions)`, and the
target is the index of the correct region.
"""

import torch.nn.functional as F

from ..heads import LinearClsHead, init_weights
from .base import Task, TaskOutput


class RefTask(Task):
    name = "ref"

    def build_head(self, backbone, config):
        head = LinearClsHead(backbone.pooled_dim, 1, pool=False)
        head.apply(init_weights)
        return head

    def metric_names(self):
        return ("accuracy",)

    def forward(self, model, batch) -> TaskOutput:
        targets = batch["target"]
        batch_size = len(targets)
        num_regions = batch["image"][0].shape[0] // batch_size

        out = model.backbone(batch, mask_text=False, mask_image=False)
        logits = model.heads[self.name](out.pooled)  # (BS*regions, 1)
        logits = logits.view(batch_size, num_regions)

        loss = F.cross_entropy(logits, targets)
        return TaskOutput(loss=loss, logits=logits, targets=targets)
