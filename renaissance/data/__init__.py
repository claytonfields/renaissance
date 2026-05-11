"""
Modern data layer for Renaissance — HF-Hub-first, with_transform-based,
batch-level collation. Lives alongside the legacy
`renaissance/datamodules/` + `renaissance/datasets/` during the migration;
once all datasets are ported the legacy directories will be deleted.
"""

from .collate import VLPCollator
from .loaders import (
    load_cc3m,
    load_cc12m,
    load_coco_karpathy,
    load_flickr30k,
    load_glue,
    load_nlvr2,
    load_refcoco,
    load_refcocog,
    load_refcocoplus,
    load_sbu,
    load_snli_ve,
    load_visual_genome,
    load_vqav2,
)
from .runner import build_collator, build_dataloader, build_dataset
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
    "load_cc3m",
    "load_cc12m",
    "load_coco_karpathy",
    "load_visual_genome",
    "load_sbu",
    "load_snli_ve",
    "load_glue",
    "build_dataset",
    "build_collator",
    "build_dataloader",
]
