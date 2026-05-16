"""NLVR2: two images, one sentence, binary judgement.

The dual-image trick is local to this task — the backbone is called
twice (`image_token_type_idx` 1 and 2) and the pooled features are
concatenated. `build_head` expands the backbone's token-type embedding
table to 3 rows (the second image uses type-id 2); this is the only
remaining backbone mutation and it's now owned by the task that needs
it, not done unconditionally in the model `__init__`.
"""

import torch
import torch.nn.functional as F

from ..heads import LinearClsHead, init_weights
from .base import Task, TaskOutput


class Nlvr2Task(Task):
    name = "nlvr2"

    def build_head(self, backbone, config):
        backbone.adjust_type_embeds_for_nlvr2()
        pooled = backbone.pooled_dim
        head = LinearClsHead(2 * pooled, 2, hidden_dim=pooled, pool=False)
        head.apply(init_weights)
        return head

    def metric_names(self):
        return ("accuracy",)

    def forward(self, model, batch) -> TaskOutput:
        out1 = model.backbone(batch, image_token_type_idx=1)
        out2 = model.backbone(batch, image_token_type_idx=2)
        fused = torch.cat([out1.pooled, out2.pooled], dim=-1)
        logits = model.heads[self.name](fused)

        labels = torch.tensor(batch["answers"], device=model.device).long()
        loss = F.cross_entropy(logits, labels.view(-1))
        return TaskOutput(loss=loss, logits=logits, targets=labels)
