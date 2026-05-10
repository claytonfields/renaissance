"""
Modern data layer for Renaissance — HF-Hub-first, with_transform-based,
batch-level collation. Lives alongside the legacy
`renaissance/datamodules/` + `renaissance/datasets/` during the migration;
once all datasets are ported the legacy directories will be deleted.
"""

from .collate import VLPCollator
from .loaders import (
    load_flickr30k,
    load_nlvr2,
    load_refcoco,
    load_refcocog,
    load_refcocoplus,
    load_vqav2,
)
from .transforms import make_image_transform, make_vlp_transform

__all__ = [
    "VLPCollator",
    "make_image_transform",
    "make_vlp_transform",
    "load_flickr30k",
    "load_vqav2",
    "load_nlvr2",
    "load_refcoco",
    "load_refcocoplus",
    "load_refcocog",
]
