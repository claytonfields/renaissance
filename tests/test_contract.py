"""
Phase 0 — modeling-layer contract tests (the safety net for the rewrite).

These tests pin the *behavioral contract* of the current `renaissance.modules`
modeling layer. They run against the current code today, and must continue
passing after the rewrite in `renaissance.modeling`. Once the rewrite is in
place, the imports of `RenaissanceTransformer` here flip over to
`RenaissanceModel`; the assertions stay verbatim.

Invariants captured:
  - `infer(batch)["cls_feats"]` shape == `(B, expected_cls_dim)` for each
    backbone × pooler configuration.
  - `forward(batch)` with `current_tasks=[]` returns infer outputs only (no
    `<task>_loss` keys).
  - For each currently-implemented task, `forward(batch)` produces a finite
    scalar `<task>_loss` and the documented `<task>_logits` shape.
  - For each task, `loss.backward()` produces at least one finite, nonzero
    gradient on the head's own parameters — i.e. gradients flow into the
    task-specific subgraph.
  - The NLVR2 head's first linear consumes a `2 * cls_dim` input (pins the
    dual-image-concat convention so the rewrite preserves it).
  - `save_pretrained` → `from_pretrained` is a bit-exact state-dict roundtrip.

Exact numerical equivalence across the rewrite is *not* asserted — init order
and RNG draws will change. Shapes, keys, finite-ness, gradient flow, and
save/load round-tripping are the contract.

Out of scope for Phase 0:
  - `ref2` runs `generalized_box_iou(preds, targets)` on a random-init MLP
    whose outputs aren't guaranteed valid boxes (x2 > x1, y2 > y1). That's a
    pre-existing model-side flaw orthogonal to the rewrite.
  - `irtr` needs `draw_false_text` negatives and a wired `rank_output` head
    derived from ITM weights; not exercised here.
  - GLUE tasks beyond `mrpc` have head classes but no `compute_*` function;
    treated as dead branches the rewrite is free to delete.
"""

import os

import pytest
import torch

from renaissance.modules import RenaissanceTransformer
from renaissance.modules.objectives import (
    compute_itm,
    compute_mlm,
    compute_mrpc,
    compute_nlvr2,
    compute_ref,
    compute_snli,
    compute_vqa,
)

from tests.conftest import BS, MAX_BB, ONE_TOWER_IMG, TWO_TOWER_IMG, VOCAB


def _finite(t):
    return torch.as_tensor(t).isfinite().all().item()


def _expected_cls_dim(config):
    """Mirror RenaissanceTransformer's hs derivation so the test fails loudly
    if the rewrite changes the documented pooled dim."""
    if config["model_type"] == "two-tower":
        return 2 * config["cross_layer_hidden_size"]
    if config["pooler_type"] == "single":
        return config["hidden_size"]
    return 2 * config["hidden_size"]


def _heads_with_grad(model, head_attr):
    """Return list of (name, grad) for params under `head_attr` that have a
    grad set. Used to assert at least one finite, nonzero grad exists."""
    head = getattr(model, head_attr)
    return [(n, p.grad) for n, p in head.named_parameters() if p.grad is not None]


def _assert_grads_flow(model, head_attr):
    grads = _heads_with_grad(model, head_attr)
    assert grads, f"no parameter under model.{head_attr} received a gradient"
    has_nonzero = False
    for name, g in grads:
        assert _finite(g), f"grad for {head_attr}.{name} is non-finite"
        if g.abs().sum().item() > 0:
            has_nonzero = True
    assert has_nonzero, f"all grads under model.{head_attr} are exactly zero"


# ---------------------------------------------------------------------------
# Infer-shape contract
# ---------------------------------------------------------------------------

class TestInferShape:

    def test_two_tower(self, two_tower_pretrain_config, two_tower_batch):
        model = RenaissanceTransformer(two_tower_pretrain_config)
        model.eval()
        with torch.no_grad():
            out = model.infer(two_tower_batch)
        assert out["cls_feats"].shape == (BS, _expected_cls_dim(two_tower_pretrain_config))
        assert _finite(out["cls_feats"])

    def test_one_tower_double_pooler(self, one_tower_pretrain_config, one_tower_batch):
        # Conftest's one-tower config uses pooler_type=double.
        assert one_tower_pretrain_config["pooler_type"] == "double"
        model = RenaissanceTransformer(one_tower_pretrain_config)
        model.eval()
        with torch.no_grad():
            out = model.infer(one_tower_batch)
        assert out["cls_feats"].shape == (BS, _expected_cls_dim(one_tower_pretrain_config))
        assert _finite(out["cls_feats"])


# ---------------------------------------------------------------------------
# `current_tasks=[]` → infer-only
# ---------------------------------------------------------------------------

def test_forward_with_no_tasks_returns_infer_outputs(two_tower_pretrain_config, two_tower_batch):
    model = RenaissanceTransformer(two_tower_pretrain_config)
    model.eval()
    model.current_tasks = []
    with torch.no_grad():
        ret = model(two_tower_batch)
    assert "cls_feats" in ret
    loss_keys = [k for k in ret if k.endswith("_loss")]
    assert loss_keys == [], f"expected no loss keys, got {loss_keys}"


# ---------------------------------------------------------------------------
# Per-task forward contract + gradient flow
# ---------------------------------------------------------------------------

class TestMlmContract:

    def test_forward_keys_and_grads(self, two_tower_pretrain_config, two_tower_batch):
        model = RenaissanceTransformer(two_tower_pretrain_config)
        model.train()
        ret = compute_mlm(model, two_tower_batch)
        assert _finite(ret["mlm_loss"])
        assert ret["mlm_loss"].ndim == 0
        ret["mlm_loss"].backward()
        _assert_grads_flow(model, "mlm_score")


class TestItmContract:

    def test_forward_keys_and_grads(self, two_tower_pretrain_config, two_tower_batch):
        model = RenaissanceTransformer(two_tower_pretrain_config)
        model.train()
        ret = compute_itm(model, two_tower_batch)
        assert _finite(ret["itm_loss"])
        assert ret["itm_logits"].shape == (BS, 2)
        ret["itm_loss"].backward()
        _assert_grads_flow(model, "itm_score")


class TestSnliContract:

    def test_forward_keys_and_grads(self, two_tower_snli_config, snli_batch):
        model = RenaissanceTransformer(two_tower_snli_config)
        model.train()
        ret = compute_snli(model, snli_batch)
        assert _finite(ret["snli_loss"])
        assert ret["snli_logits"].shape == (BS, 3)
        ret["snli_loss"].backward()
        _assert_grads_flow(model, "snli_classifier")


class TestNlvr2Contract:

    def test_forward_keys_and_grads(self, two_tower_nlvr2_config, nlvr2_batch):
        model = RenaissanceTransformer(two_tower_nlvr2_config)
        model.train()
        ret = compute_nlvr2(model, nlvr2_batch)
        assert _finite(ret["nlvr2_loss"])
        assert ret["nlvr2_logits"].shape == (BS, 2)
        ret["nlvr2_loss"].backward()
        _assert_grads_flow(model, "nlvr2_classifier")

    def test_head_consumes_concatenated_features(self, two_tower_nlvr2_config):
        """NLVR2 fuses two image streams by concatenating their cls features —
        pin the head input dim so the rewrite preserves the convention."""
        model = RenaissanceTransformer(two_tower_nlvr2_config)
        cls_dim = _expected_cls_dim(two_tower_nlvr2_config)
        assert model.nlvr2_classifier.dense.in_features == 2 * cls_dim


class TestVqaContract:

    def test_forward_keys_and_grads(self, two_tower_vqa_config, vqa_batch):
        model = RenaissanceTransformer(two_tower_vqa_config)
        model.train()
        ret = compute_vqa(model, vqa_batch)
        assert _finite(ret["vqa_loss"])
        n_labels = two_tower_vqa_config["vqav2_label_size"]
        assert ret["vqa_logits"].shape == (BS, n_labels)
        ret["vqa_loss"].backward()
        _assert_grads_flow(model, "vqa_classifier")


class TestRefContract:

    def test_forward_keys_and_grads(self, two_tower_ref_config, ref_batch):
        model = RenaissanceTransformer(two_tower_ref_config)
        model.train()
        ret = compute_ref(model, ref_batch)
        assert _finite(ret["ref_loss"])
        assert ret["ref_logits"].shape == (BS, MAX_BB)
        ret["ref_loss"].backward()
        _assert_grads_flow(model, "ref_classifier")


class TestMrpcContract:

    def test_forward_keys_and_grads(self, two_tower_mrpc_config, mrpc_batch):
        model = RenaissanceTransformer(two_tower_mrpc_config)
        model.train()
        ret = compute_mrpc(model, mrpc_batch)
        assert _finite(ret["mrpc_loss"])
        assert ret["mrpc_logits"].shape == (BS, 2)
        ret["mrpc_loss"].backward()
        _assert_grads_flow(model, "mrpc_classifier")


# ---------------------------------------------------------------------------
# save_pretrained / from_pretrained roundtrip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(two_tower_snli_config, tmp_path):
    """Pin: persisted state is exactly recoverable. Bumps when the rewrite
    changes param naming — that's the signal to update the rewrite's
    from_pretrained shim (or to bump the model version) deliberately."""
    model = RenaissanceTransformer(two_tower_snli_config)
    model.eval()
    out_dir = tmp_path / "ckpt"
    model.save_pretrained(str(out_dir))
    assert (out_dir / "model.safetensors").exists()

    restored = RenaissanceTransformer.from_pretrained(str(out_dir))
    original_state = model.state_dict()
    restored_state = restored.state_dict()
    assert set(original_state.keys()) == set(restored_state.keys())
    for k in original_state:
        assert torch.equal(original_state[k], restored_state[k]), f"mismatch on {k}"
