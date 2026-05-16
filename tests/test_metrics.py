"""
Phase 6 — `renaissance.modeling.metrics.TaskMetrics`.

Pins the model-owned metrics subsystem that replaces
`renaissance_utils.set_metrics` + `epoch_wrapup`: it builds the right
objects from each task's `metric_names()`, updates from a `TaskOutput`,
and compute()-then-resets per epoch.
"""

import torch

from renaissance.modeling.metrics import TaskMetrics
from renaissance.modeling.tasks import TASK_REGISTRY, TaskOutput


def _out(loss, logits=None, targets=None):
    return TaskOutput(loss=torch.tensor(loss), logits=logits, targets=targets)


def test_builds_loss_plus_declared_metrics():
    tm = TaskMetrics(["mlm", "vqa", "mrpc", "ref2"], TASK_REGISTRY, phases=("train", "val"))
    keys = set(tm._metrics.keys())
    # loss is always present per (phase, task)
    assert "train__mlm__loss" in keys and "val__mlm__loss" in keys
    # declared extras: vqa→score, mrpc→accuracy+f1, ref2→iou
    assert "train__vqa__score" in keys
    assert "train__mrpc__f1" in keys and "train__mrpc__accuracy" in keys
    assert "train__ref2__iou" in keys


def test_update_and_compute_then_reset():
    tm = TaskMetrics(["snli"], TASK_REGISTRY, phases=("train", "val"))
    logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    targets = torch.tensor([0, 1])  # both correct
    tm.update("train", "snli", _out(1.5, logits, targets))

    m = tm.compute("train")
    assert m["snli/train/loss_epoch"] == 1.5
    assert m["snli/train/accuracy_epoch"] == 1.0

    # compute() reset the state — a fresh epoch starts from zero counts.
    tm.update("train", "snli", _out(0.5, logits, torch.tensor([1, 0])))  # both wrong
    m2 = tm.compute("train")
    assert m2["snli/train/loss_epoch"] == 0.5
    assert m2["snli/train/accuracy_epoch"] == 0.0


def test_phase_isolation():
    tm = TaskMetrics(["itm"], TASK_REGISTRY, phases=("train", "val"))
    tm.update("train", "itm", _out(3.0, torch.tensor([[1.0, 0.0]]), torch.tensor([0])))
    # val never updated → val loss metric computes its empty default, train carries the value
    assert tm.compute("train")["itm/train/loss_epoch"] == 3.0


def test_unknown_phase_is_noop():
    tm = TaskMetrics(["itm"], TASK_REGISTRY, phases=("train",))
    tm.update("val", "itm", _out(9.0))  # phase not built → ignored, no raise
    assert "itm/train/loss_epoch" in tm.compute("train")
