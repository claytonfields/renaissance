"""Image-text matching with in-batch negatives."""

import torch
import torch.nn.functional as F

from ..heads import ItmHead, init_weights
from .base import Task, TaskOutput


class ItmTask(Task):
    name = "itm"

    def build_head(self, backbone, config):
        head = ItmHead(backbone.pooled_dim)
        head.apply(init_weights)
        return head

    def metric_names(self):
        return ("accuracy",)

    def forward(self, model, batch) -> TaskOutput:
        pos_len = len(batch["text"]) // 2
        neg_len = len(batch["text"]) - pos_len
        labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)]).to(model.device)
        labels = labels[torch.randperm(labels.size(0))]

        itm_images = [
            torch.stack(
                [ti if labels[i] == 1 else fi for i, (ti, fi) in enumerate(zip(bti, bfi))]
            )
            for bti, bfi in zip(batch["image"], batch["false_image_0"])
        ]
        batch = {**batch, "image": itm_images}

        out = model.backbone(batch, mask_text=False, mask_image=False)
        logits = model.heads[self.name](out.pooled)
        loss = F.cross_entropy(logits, labels.long())
        return TaskOutput(loss=loss, logits=logits, targets=labels)
