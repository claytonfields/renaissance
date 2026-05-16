"""
Phase 3 — unit tests for `renaissance.modeling.tasks`.

Exercises each Task through the *new* stack (new backbone + new heads +
new task) using a lightweight stand-in for the Phase 4 model: an object
exposing `.backbone`, `.heads`, `.config`, `.device`. This mirrors the
Phase 0 contract (loss finite + scalar, logits shape, gradient flows into
the head) but proves the rewritten path produces it.

ref2 is forward-excluded for the same reason as Phase 0 — its
unconstrained-MLP output isn't a valid box for `generalized_box_iou`. We
still check its head builds with width 4.
"""

import pytest
import torch
import torch.nn as nn

from renaissance.modeling.backbones import build_backbone
from renaissance.modeling.tasks import TASK_REGISTRY, get_task

from tests.conftest import BS, MAX_BB


class _Harness:
    """Minimal Phase-4-model stand-in: just the surface tasks touch."""

    def __init__(self, config):
        self.config = config
        self.backbone = build_backbone(config)
        self.heads = nn.ModuleDict()

    @property
    def device(self):
        return next(self.backbone.parameters()).device

    def add(self, task):
        head = task.build_head(self.backbone, self.config)
        if head is not None:
            self.heads[task.name] = head
        return head


def _finite(t):
    return torch.as_tensor(t).isfinite().all().item()


def _assert_head_grads(head):
    grads = [p.grad for p in head.parameters() if p.grad is not None]
    assert grads, "no gradient reached the head"
    assert any(g.abs().sum().item() > 0 for g in grads), "all head grads zero"
    assert all(_finite(g) for g in grads), "non-finite head grad"


def _run(config, task_name, batch):
    h = _Harness(config)
    task = get_task(task_name)
    h.add(task)
    out = task.forward(h, batch)
    return out, h


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_contents():
    assert set(TASK_REGISTRY) == {
        "mlm", "itm", "vqa", "nlvr2", "snli", "ref", "ref2", "mrpc"
    }
    for name, task in TASK_REGISTRY.items():
        assert task.name == name


def test_get_task_unknown_raises():
    with pytest.raises(ValueError, match="Unknown task"):
        get_task("bogus")


# ---------------------------------------------------------------------------
# Per-task forward + gradient flow (new stack)
# ---------------------------------------------------------------------------

def test_mlm(two_tower_pretrain_config, two_tower_batch):
    out, _ = _run(two_tower_pretrain_config, "mlm", two_tower_batch)
    assert out.loss.ndim == 0 and _finite(out.loss)
    out.loss.backward()


def test_itm(two_tower_pretrain_config, two_tower_batch):
    out, h = _run(two_tower_pretrain_config, "itm", two_tower_batch)
    assert out.logits.shape == (BS, 2)
    assert _finite(out.loss)
    out.loss.backward()
    _assert_head_grads(h.heads["itm"])


def test_vqa(two_tower_vqa_config, vqa_batch):
    out, h = _run(two_tower_vqa_config, "vqa", vqa_batch)
    n = two_tower_vqa_config["vqav2_label_size"]
    assert out.logits.shape == (BS, n)
    assert out.targets.shape == (BS, n)
    assert _finite(out.loss)
    assert out.extras["vqa_labels"] == vqa_batch["vqa_labels"]
    out.loss.backward()
    _assert_head_grads(h.heads["vqa"])


def test_nlvr2(two_tower_nlvr2_config, nlvr2_batch):
    out, h = _run(two_tower_nlvr2_config, "nlvr2", nlvr2_batch)
    assert out.logits.shape == (BS, 2)
    assert _finite(out.loss)
    out.loss.backward()
    _assert_head_grads(h.heads["nlvr2"])


def test_snli(two_tower_snli_config, snli_batch):
    out, h = _run(two_tower_snli_config, "snli", snli_batch)
    assert out.logits.shape == (BS, 3)
    assert _finite(out.loss)
    out.loss.backward()
    _assert_head_grads(h.heads["snli"])


def test_ref(two_tower_ref_config, ref_batch):
    out, h = _run(two_tower_ref_config, "ref", ref_batch)
    assert out.logits.shape == (BS, MAX_BB)
    assert _finite(out.loss)
    out.loss.backward()
    _assert_head_grads(h.heads["ref"])


def test_mrpc(two_tower_mrpc_config, mrpc_batch):
    out, h = _run(two_tower_mrpc_config, "mrpc", mrpc_batch)
    assert out.logits.shape == (BS, 2)
    assert _finite(out.loss)
    out.loss.backward()
    _assert_head_grads(h.heads["mrpc"])


def test_ref2_head_only(two_tower_pretrain_config):
    # Forward excluded (see module docstring); pin the head width.
    h = _Harness(two_tower_pretrain_config)
    head = h.add(get_task("ref2"))
    pooled = torch.randn(BS, h.backbone.pooled_dim)
    assert head(pooled).shape == (BS, 4)
