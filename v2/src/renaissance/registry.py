"""One registry mechanism for every component kind.

Built-in components self-register at import time via the ``@register``
decorator. Third-party packages extend Renaissance without forking it by
exposing entry points in the ``renaissance.components`` group, named
``"<kind>.<name>"``:

    [project.entry-points."renaissance.components"]
    "fusion.my-fusion" = "my_pkg.fusion:MyFusion"

``get()`` falls back to that group on a registry miss, so an installed
extension is loadable the first time a config references it.
"""

from __future__ import annotations

from importlib.metadata import entry_points

KINDS = ("encoder", "fusion", "head", "task")
ENTRY_POINT_GROUP = "renaissance.components"

_REGISTRY: dict[str, dict[str, type]] = {kind: {} for kind in KINDS}


def register(kind: str, name: str):
    """Class decorator: make ``cls`` available as ``get(kind, name)``.

    The class must carry a ``Config`` attribute (its component-owned config
    dataclass) — that contract is what lets the global schema be composed
    from the registries.
    """
    _check_kind(kind)

    def deco(cls: type) -> type:
        if name in _REGISTRY[kind]:
            raise ValueError(f"Duplicate {kind} name {name!r} (already {_REGISTRY[kind][name].__qualname__})")
        if not hasattr(cls, "Config"):
            raise TypeError(f"{cls.__qualname__} must define a Config dataclass to be registered")
        cls.registry_kind = kind
        cls.registry_name = name
        _REGISTRY[kind][name] = cls
        return cls

    return deco


def get(kind: str, name: str) -> type:
    """Look up a registered component class, loading entry-point extensions on miss."""
    _check_kind(kind)
    if name not in _REGISTRY[kind]:
        _load_entry_point(kind, name)
    try:
        return _REGISTRY[kind][name]
    except KeyError:
        raise KeyError(f"Unknown {kind} {name!r}. Available: {available(kind)}") from None


def available(kind: str) -> list[str]:
    _check_kind(kind)
    return sorted(_REGISTRY[kind])


def _check_kind(kind: str) -> None:
    if kind not in KINDS:
        raise ValueError(f"Unknown component kind {kind!r}. Kinds: {list(KINDS)}")


def _load_entry_point(kind: str, name: str) -> None:
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.name == f"{kind}.{name}":
            cls = ep.load()
            if name not in _REGISTRY[kind]:  # the extension may self-register on import
                register(kind, name)(cls)
            return
