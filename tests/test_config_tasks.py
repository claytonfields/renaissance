"""
Phase 5 — `tasks` list + `task_config`, with the `loss_names` shim.

Pins: `normalize_tasks` precedence/derivation both directions, the
omegaconf + to_flat_dict paths normalize, `RenaissanceModel` honors a
`tasks` list, and the hub config round-trips the new fields.
"""

import pytest

from renaissance.config_schema import (
    ALL_LOSS_NAMES,
    RenaissanceConfig,
    from_omegaconf,
    normalize_tasks,
    to_flat_dict,
)
from renaissance.hub import RenaissanceHubConfig
from renaissance.modeling import RenaissanceModel


# ---------------------------------------------------------------------------
# normalize_tasks
# ---------------------------------------------------------------------------

class TestNormalizeTasks:

    def test_tasks_derives_loss_names(self):
        out = normalize_tasks({"tasks": ["mlm", "itm"]})
        assert out["loss_names"]["mlm"] == 1
        assert out["loss_names"]["itm"] == 1
        assert out["loss_names"]["vqa"] == 0
        assert set(out["tasks"]) == {"mlm", "itm"}

    def test_loss_names_derives_tasks(self):
        out = normalize_tasks({"loss_names": {"snli": 1}})
        assert out["tasks"] == ["snli"]
        # full table is filled in for downstream consumers (data runner etc.)
        assert out["loss_names"]["snli"] == 1
        assert out["loss_names"]["mlm"] == 0

    def test_tasks_wins_over_loss_names(self):
        out = normalize_tasks({"tasks": ["vqa"], "loss_names": {"mlm": 1, "itm": 1}})
        assert out["tasks"] == ["vqa"]
        assert out["loss_names"]["vqa"] == 1
        assert out["loss_names"]["mlm"] == 0
        assert out["loss_names"]["itm"] == 0

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task name"):
            normalize_tasks({"tasks": ["mlm", "bogus"]})

    def test_idempotent(self):
        once = normalize_tasks({"tasks": ["mlm", "itm"]})
        twice = normalize_tasks(dict(once))
        assert once["tasks"] == twice["tasks"]
        assert once["loss_names"] == twice["loss_names"]

    def test_deterministic_task_order(self):
        # Order follows ALL_LOSS_NAMES regardless of input order.
        out = normalize_tasks({"tasks": ["itm", "mlm"]})
        expected = [t for t in ALL_LOSS_NAMES if t in {"itm", "mlm"}]
        assert out["tasks"] == expected


# ---------------------------------------------------------------------------
# Conversion paths normalize
# ---------------------------------------------------------------------------

def test_to_flat_dict_normalizes_default():
    flat = to_flat_dict(RenaissanceConfig())
    # Default TaskConfig.loss_names activates itm+mlm.
    assert set(flat["tasks"]) == {"itm", "mlm"}
    assert flat["loss_names"]["mlm"] == 1


def test_from_omegaconf_flat_with_tasks():
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"model_type": "two-tower", "tasks": ["snli"]})
    flat = from_omegaconf(cfg)
    assert flat["tasks"] == ["snli"]
    assert flat["loss_names"]["snli"] == 1


def test_from_omegaconf_grouped_loss_names_shim():
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"task": {"loss_names": {"vqa": 1}}})
    flat = from_omegaconf(cfg)
    assert flat["tasks"] == ["vqa"]


# ---------------------------------------------------------------------------
# RenaissanceModel honors the tasks list
# ---------------------------------------------------------------------------

def test_model_uses_tasks_list(two_tower_pretrain_config):
    cfg = dict(two_tower_pretrain_config)
    cfg.pop("loss_names", None)  # only the tasks list present
    cfg["tasks"] = ["snli"]
    model = RenaissanceModel(cfg)
    assert set(model.heads.keys()) == {"snli"}


def test_model_still_works_with_loss_names_only(two_tower_snli_config):
    # Conftest configs have no `tasks` key — the fallback path must hold.
    assert "tasks" not in two_tower_snli_config
    model = RenaissanceModel(two_tower_snli_config)
    assert set(model.heads.keys()) == {"snli"}


# ---------------------------------------------------------------------------
# Hub round-trip
# ---------------------------------------------------------------------------

def test_hub_roundtrips_tasks_and_task_config():
    flat = normalize_tasks({
        "model_type": "two-tower",
        "tasks": ["vqa"],
        "task_config": {"vqa": {"label_size": 3129}},
    })
    hub = RenaissanceHubConfig.from_flat_config(flat)
    back = hub.to_flat_config()
    assert back["tasks"] == ["vqa"]
    assert back["task_config"] == {"vqa": {"label_size": 3129}}
    assert back["loss_names"]["vqa"] == 1
