"""
Modern data layer for Renaissance — HF-Hub-first, with_transform-based,
batch-level collation. Lives alongside the legacy
`renaissance/datamodules/` + `renaissance/datasets/` during the migration;
once all datasets are ported the legacy directories will be deleted.
"""

from .collate import VLPCollator
from .transforms import make_image_transform, make_vlp_transform
from .loaders import load_flickr30k

__all__ = [
    "VLPCollator",
    "make_image_transform",
    "make_vlp_transform",
    "load_flickr30k",
]
