"""Referring-expression resolution as bbox regression (GIoU loss).

Targets are normalized xyxy boxes from the modern data layer
(`batch["bbox"]`). Caveat preserved from the legacy code: the head is an
unconstrained MLP, so its raw 4-vectors aren't guaranteed valid boxes
(x2 > x1, y2 > y1) for `generalized_box_iou` — a model-side issue
tracked separately, not introduced here.
"""

import torch
from torchvision.ops import generalized_box_iou

from ..heads import LinearClsHead, init_weights
from .base import Task, TaskOutput


class Ref2Task(Task):
    name = "ref2"

    def build_head(self, backbone, config):
        head = LinearClsHead(backbone.pooled_dim, 4, pool=False)
        head.apply(init_weights)
        return head

    def metric_names(self):
        return ("iou",)

    def forward(self, model, batch) -> TaskOutput:
        targets = batch["bbox"]
        if not torch.is_tensor(targets):
            targets = torch.tensor(targets, dtype=torch.float32)
        targets = targets.to(model.device).float()

        out = model.backbone(batch, mask_text=False, mask_image=False)
        preds = model.heads[self.name](out.pooled)

        loss = (1 - generalized_box_iou(preds, targets).diag()).mean()
        return TaskOutput(loss=loss, logits=preds, targets=targets)
