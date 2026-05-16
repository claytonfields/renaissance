"""
Model-owned metrics (Phase 6).

Replaces `renaissance_utils.set_metrics` + `epoch_wrapup` — ~150 lines of
string-munged `setattr(pl_module, f"{split}_{k}_accuracy", ...)` and a
giant per-task if-ladder that recomputed/reset by name.

`TaskMetrics` builds one metric object per (phase, task, metric) from
each task's declared `metric_names()` plus an always-present `loss`
scalar, updates them from a `TaskOutput`, and computes+resets per epoch.
Adding a task needs nothing here — its `metric_names()` drives the rest.

Phases are train/val only. The legacy dev/test split routing for
snli/nlvr2 and the irtr-recall path were tied to the legacy data layer's
`table_name` convention; deferred (irtr isn't a registered task anyway).
"""

import torch.nn as nn
from torchmetrics.classification import BinaryF1Score

from renaissance.gadgets.my_metrics import IoU, Accuracy, Scalar, VQAScore

# Metric-name (as declared by Task.metric_names) → constructor.
_METRIC_FACTORY = {
    "accuracy": Accuracy,
    "score": VQAScore,
    "iou": IoU,
    "f1": BinaryF1Score,
}


def _key(phase, task, metric):
    # nn.ModuleDict keys may not contain ".".
    return f"{phase}__{task}__{metric}"


class TaskMetrics(nn.Module):
    def __init__(self, task_names, registry, phases=("train", "val")):
        super().__init__()
        self.phases = tuple(phases)
        self._metrics = nn.ModuleDict()
        # task -> tuple of extra metric names (beyond loss)
        self._task_metrics = {}
        for name in task_names:
            extra = tuple(registry[name].metric_names())
            self._task_metrics[name] = extra
            for phase in self.phases:
                self._metrics[_key(phase, name, "loss")] = Scalar()
                for m in extra:
                    self._metrics[_key(phase, name, m)] = _METRIC_FACTORY[m]()

    def update(self, phase, task_name, out):
        if phase not in self.phases:
            return
        self._metrics[_key(phase, task_name, "loss")].update(out.loss)
        for m in self._task_metrics.get(task_name, ()):
            metric = self._metrics[_key(phase, task_name, m)]
            if m == "f1":
                metric.update(out.logits.argmax(dim=-1), out.targets)
            else:
                metric.update(out.logits, out.targets)

    def compute(self, phase) -> dict:
        """Compute + reset every metric for `phase`. Returns
        ``{"<task>/<phase>/<metric>_epoch": float}``."""
        result = {}
        for name, extra in self._task_metrics.items():
            for m in ("loss", *extra):
                metric = self._metrics[_key(phase, name, m)]
                value = metric.compute()
                result[f"{name}/{phase}/{m}_epoch"] = (
                    value.item() if hasattr(value, "item") else float(value)
                )
                metric.reset()
        return result
