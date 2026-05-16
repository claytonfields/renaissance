"""SNLI visual entailment (3-way classification).

Phase 3 ports the training-path math only. The legacy dev/test split
routing in `compute_snli` is an eval-time metric concern handled in
Phase 6.
"""

import torch
import torch.nn.functional as F

from ..heads import LinearClsHead, init_weights
from .base import Task, TaskOutput


class SnliTask(Task):
    name = "snli"

    def build_head(self, backbone, config):
        head = LinearClsHead(backbone.pooled_dim, 3, pool=False)
        head.apply(init_weights)
        return head

    def metric_names(self):
        return ("accuracy",)

    def forward(self, model, batch) -> TaskOutput:
        out = model.backbone(batch, mask_text=False, mask_image=False)
        logits = model.heads[self.name](out.pooled)
        labels = torch.tensor(batch["labels"], device=model.device).long()
        loss = F.cross_entropy(logits, labels.view(-1))
        return TaskOutput(loss=loss, logits=logits, targets=labels)
