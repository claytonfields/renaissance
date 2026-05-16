"""
Phase 4 — `RenaissanceModel` satisfies the Phase 0 behavioral contract.

These mirror `test_contract.py` (which still guards the legacy model)
but run against the rewritten `renaissance.modeling.RenaissanceModel`.
When Phase 7 retires the legacy model, `test_contract.py` flips its
import here and these become redundant; until then they prove the new
model is contract-equivalent.
"""

import pytest
import torch

from renaissance.modeling import RenaissanceModel
from renaissance.modules import renaissance_utils

from tests.conftest import BS, MAX_BB


def _finite(t):
    return torch.as_tensor(t).isfinite().all().item()


def _expected_cls_dim(config):
    if config["model_type"] == "two-tower":
        return 2 * config["cross_layer_hidden_size"]
    if config["pooler_type"] == "single":
        return config["hidden_size"]
    return 2 * config["hidden_size"]


def _assert_head_grads(model, name):
    grads = [p.grad for p in model.heads[name].parameters() if p.grad is not None]
    assert grads, f"no gradient reached model.heads[{name!r}]"
    assert any(g.abs().sum().item() > 0 for g in grads), "all head grads zero"
    assert all(_finite(g) for g in grads), "non-finite head grad"


def _run_task(model, name, batch):
    model.train()
    model.current_tasks = [name]
    return model(batch)


# ---------------------------------------------------------------------------
# Infer shape
# ---------------------------------------------------------------------------

def test_two_tower_infer_shape(two_tower_pretrain_config, two_tower_batch):
    model = RenaissanceModel(two_tower_pretrain_config)
    model.eval()
    with torch.no_grad():
        out = model.infer(two_tower_batch)
    assert out["cls_feats"].shape == (BS, _expected_cls_dim(two_tower_pretrain_config))
    assert out.pooled.shape[-1] == model.backbone.pooled_dim


def test_one_tower_infer_shape(one_tower_pretrain_config, one_tower_batch):
    model = RenaissanceModel(one_tower_pretrain_config)
    model.eval()
    with torch.no_grad():
        out = model.infer(one_tower_batch)
    assert out["cls_feats"].shape == (BS, _expected_cls_dim(one_tower_pretrain_config))


# ---------------------------------------------------------------------------
# Forward with no tasks → infer-only
# ---------------------------------------------------------------------------

def test_forward_no_tasks_is_infer_only(two_tower_pretrain_config, two_tower_batch):
    model = RenaissanceModel(two_tower_pretrain_config)
    model.eval()
    model.current_tasks = []
    with torch.no_grad():
        ret = model(two_tower_batch)
    assert "cls_feats" in ret
    assert [k for k in ret if k.endswith("_loss")] == []


# ---------------------------------------------------------------------------
# Per-task forward keys + gradient flow
# ---------------------------------------------------------------------------

def test_mlm(two_tower_pretrain_config, two_tower_batch):
    model = RenaissanceModel(two_tower_pretrain_config)
    ret = _run_task(model, "mlm", two_tower_batch)
    assert _finite(ret["mlm_loss"]) and ret["mlm_loss"].ndim == 0
    ret["mlm_loss"].backward()
    _assert_head_grads(model, "mlm")


def test_itm(two_tower_pretrain_config, two_tower_batch):
    model = RenaissanceModel(two_tower_pretrain_config)
    ret = _run_task(model, "itm", two_tower_batch)
    assert ret["itm_logits"].shape == (BS, 2)
    assert _finite(ret["itm_loss"])
    ret["itm_loss"].backward()
    _assert_head_grads(model, "itm")


def test_snli(two_tower_snli_config, snli_batch):
    model = RenaissanceModel(two_tower_snli_config)
    ret = _run_task(model, "snli", snli_batch)
    assert ret["snli_logits"].shape == (BS, 3)
    assert _finite(ret["snli_loss"])
    ret["snli_loss"].backward()
    _assert_head_grads(model, "snli")


def test_nlvr2(two_tower_nlvr2_config, nlvr2_batch):
    model = RenaissanceModel(two_tower_nlvr2_config)
    ret = _run_task(model, "nlvr2", nlvr2_batch)
    assert ret["nlvr2_logits"].shape == (BS, 2)
    assert _finite(ret["nlvr2_loss"])
    ret["nlvr2_loss"].backward()
    _assert_head_grads(model, "nlvr2")


def test_nlvr2_head_consumes_concat(two_tower_nlvr2_config):
    model = RenaissanceModel(two_tower_nlvr2_config)
    cls_dim = _expected_cls_dim(two_tower_nlvr2_config)
    assert model.heads["nlvr2"].dense.in_features == 2 * cls_dim


def test_vqa(two_tower_vqa_config, vqa_batch):
    model = RenaissanceModel(two_tower_vqa_config)
    ret = _run_task(model, "vqa", vqa_batch)
    n = two_tower_vqa_config["vqav2_label_size"]
    assert ret["vqa_logits"].shape == (BS, n)
    assert _finite(ret["vqa_loss"])
    ret["vqa_loss"].backward()
    _assert_head_grads(model, "vqa")


def test_ref(two_tower_ref_config, ref_batch):
    model = RenaissanceModel(two_tower_ref_config)
    ret = _run_task(model, "ref", ref_batch)
    assert ret["ref_logits"].shape == (BS, MAX_BB)
    assert _finite(ret["ref_loss"])
    ret["ref_loss"].backward()
    _assert_head_grads(model, "ref")


def test_mrpc(two_tower_mrpc_config, mrpc_batch):
    model = RenaissanceModel(two_tower_mrpc_config)
    ret = _run_task(model, "mrpc", mrpc_batch)
    assert ret["mrpc_logits"].shape == (BS, 2)
    assert _finite(ret["mrpc_loss"])
    ret["mrpc_loss"].backward()
    _assert_head_grads(model, "mrpc")


# ---------------------------------------------------------------------------
# Active-set is config-driven
# ---------------------------------------------------------------------------

def test_only_active_heads_built(two_tower_snli_config):
    model = RenaissanceModel(two_tower_snli_config)
    assert set(model.heads.keys()) == {"snli"}


# ---------------------------------------------------------------------------
# save_pretrained / from_pretrained roundtrip
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Phase 6: set_active_tasks / epoch_metrics / set_schedule LR grouping
# ---------------------------------------------------------------------------

def test_set_active_tasks(two_tower_pretrain_config):
    model = RenaissanceModel(two_tower_pretrain_config)
    assert model.current_tasks == []
    model.set_active_tasks()
    assert set(model.current_tasks) == {"mlm", "itm"}


def test_forward_populates_epoch_metrics(two_tower_snli_config, snli_batch):
    model = RenaissanceModel(two_tower_snli_config)
    model.train()
    model.current_tasks = ["snli"]
    model(snli_batch)
    m = model.epoch_metrics("train")
    assert "snli/train/loss_epoch" in m
    assert "snli/train/accuracy_epoch" in m
    # computed → reset; a second compute with no updates restarts cleanly
    assert "snli/train/loss_epoch" in model.epoch_metrics("train")


def test_set_schedule_gives_heads_the_head_lr(two_tower_snli_config):
    model = RenaissanceModel(two_tower_snli_config)
    optimizer, _ = renaissance_utils.set_schedule(
        model, two_tower_snli_config, max_steps=10
    )
    lr = two_tower_snli_config["learning_rate"]
    head_lr = lr * two_tower_snli_config["lr_mult_head"]
    head_param_ids = {
        id(p) for n, p in model.named_parameters() if n.startswith("heads.")
    }
    assert head_param_ids, "no heads.* params found"
    # The warmup scheduler scales param_groups["lr"] to ~0 at step 0; the
    # group's configured base rate is preserved in "initial_lr".
    placed = {
        id(p)
        for g in optimizer.param_groups
        if abs(g.get("initial_lr", g["lr"]) - head_lr) < 1e-12
        for p in g["params"]
    }
    assert head_param_ids.issubset(placed), "head params not on the head LR"


def test_save_load_roundtrip(two_tower_snli_config, tmp_path):
    model = RenaissanceModel(two_tower_snli_config)
    model.eval()
    out_dir = tmp_path / "ckpt"
    model.save_pretrained(str(out_dir))
    assert (out_dir / "model.safetensors").exists()

    restored = RenaissanceModel.from_pretrained(str(out_dir))
    a, b = model.state_dict(), restored.state_dict()
    assert set(a) == set(b)
    for k in a:
        assert torch.equal(a[k], b[k]), f"mismatch on {k}"
