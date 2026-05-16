"""MRPC (GLUE) — text-only paraphrase classification.

The only GLUE task with a real `compute_*` in the legacy code; the other
GLUE names had heads but no objective (dead branches the rewrite drops).
Uses the backbone's text-only path; the head pools the CLS token.
"""

import torch.nn.functional as F

from ..heads import LinearClsHead, init_weights
from .base import Task, TaskOutput


class MrpcTask(Task):
    name = "mrpc"

    def build_head(self, backbone, config):
        head = LinearClsHead(backbone.text_hidden_size, 2, pool=True)
        head.apply(init_weights)
        return head

    def metric_names(self):
        return ("accuracy", "f1")

    def forward(self, model, batch) -> TaskOutput:
        labels = batch.pop("label", None)
        hidden = model.backbone.encode_text_only(batch)
        logits = model.heads[self.name](hidden)
        loss = F.cross_entropy(logits, labels)
        return TaskOutput(loss=loss, logits=logits, targets=labels)
