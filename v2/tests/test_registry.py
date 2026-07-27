"""Unit tests for the component registry."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from renaissance import registry


@dataclass
class _Cfg:
    x: int = 0


@pytest.fixture(autouse=True)
def _clean_registry():
    """Snapshot and restore the registry around every test."""
    snapshot = {kind: dict(registry._REGISTRY[kind]) for kind in registry.KINDS}
    yield
    for kind in registry.KINDS:
        registry._REGISTRY[kind].clear()
        registry._REGISTRY[kind].update(snapshot[kind])


def test_register_and_get():
    @registry.register("fusion", "test-fusion")
    class F:
        Config = _Cfg

    assert registry.get("fusion", "test-fusion") is F
    assert F.registry_kind == "fusion"
    assert F.registry_name == "test-fusion"
    assert "test-fusion" in registry.available("fusion")


def test_duplicate_name_rejected():
    @registry.register("head", "test-head")
    class H1:
        Config = _Cfg

    with pytest.raises(ValueError, match="Duplicate head name 'test-head'"):

        @registry.register("head", "test-head")
        class H2:
            Config = _Cfg


def test_config_attribute_required():
    with pytest.raises(TypeError, match="must define a Config"):

        @registry.register("task", "test-task")
        class T:
            pass


def test_unknown_kind():
    with pytest.raises(ValueError, match="Unknown component kind 'optimizer'"):
        registry.get("optimizer", "adamw")


def test_unknown_name_lists_available():
    with pytest.raises(KeyError, match="Unknown fusion 'nope'"):
        registry.get("fusion", "nope")


def test_entry_point_lazy_load(monkeypatch):
    class Ext:
        Config = _Cfg

    ep = SimpleNamespace(name="encoder.ext-encoder", load=lambda: Ext)
    monkeypatch.setattr(registry, "entry_points", lambda group: [ep] if group == registry.ENTRY_POINT_GROUP else [])

    assert registry.get("encoder", "ext-encoder") is Ext
    # second lookup hits the registry, not the entry point
    assert registry.get("encoder", "ext-encoder") is Ext
