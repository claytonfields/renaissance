"""Visual question answering (soft-target BCE, BAN-VQA scaling)."""

import torch
import torch.nn.functional as F

from ..heads import LinearClsHead, init_weights
from .base import Task, TaskOutput


class VqaTask(Task):
    name = "vqa"

    def build_head(self, backbone, config):
        head = LinearClsHead(
            backbone.pooled_dim, config["vqav2_label_size"], pool=False
        )
        head.apply(init_weights)
        return head

    def metric_names(self):
        return ("score",)

    def forward(self, model, batch) -> TaskOutput:
        out = model.backbone(batch, mask_text=False, mask_image=False)
        logits = model.heads[self.name](out.pooled)

        n_labels = model.config["vqav2_label_size"]
        targets = torch.zeros(len(logits), n_labels, device=model.device)
        for i, (labs, scores) in enumerate(
            zip(batch["vqa_labels"], batch["vqa_scores"])
        ):
            for label, score in zip(labs, scores):
                targets[i, label] = score

        # BAN-VQA: scale BCE by the label count.
        loss = F.binary_cross_entropy_with_logits(logits, targets) * targets.shape[1]
        return TaskOutput(
            loss=loss,
            logits=logits,
            targets=targets,
            extras={
                "vqa_labels": batch["vqa_labels"],
                "vqa_scores": batch["vqa_scores"],
            },
        )
