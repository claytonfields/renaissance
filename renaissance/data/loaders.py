"""
Per-dataset loaders. Each function returns a `datasets.Dataset` (or
`IterableDataset` for streaming sources) with `with_transform` applied so
each row yields `{image: Tensor[3, H, W], text: str, ...}` (or
`{image_0, image_1, text, ...}` for two-image datasets).

Pair these with `VLPCollator` and a standard `torch.utils.data.DataLoader`.

Dataset-specific label processing (VQA answer-vocab mapping, RefCOCO bbox
normalization, etc.) is deferred to the evaluator / objective so loaders
stay thin.
"""

from typing import Optional

from datasets import Image as DSImage
from datasets import load_dataset

from .transforms import make_vlp_transform


def load_flickr30k(
    split: str,
    image_size: int,
    hub_id: str = "nlphuji/flickr30k",
    seed: Optional[int] = None,
):
    """Karpathy-split Flickr30k from `nlphuji/flickr30k`.

    All 31k images live in a single "test" split with a `split` column
    tagging Karpathy assignment. Each row has a `caption` column with 5
    strings.
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be one of train/val/test, got {split!r}")

    ds = load_dataset(hub_id, split="test")
    ds = ds.filter(lambda x: x["split"] == split)
    ds = ds.cast_column("image", DSImage(decode=True))
    ds = ds.with_transform(
        make_vlp_transform(
            image_size,
            image_columns={"image": "image"},
            text_column="caption",
            multi_caption=True,
            seed=seed,
        )
    )
    return ds


def load_vqav2(
    split: str,
    image_size: int,
    hub_id: str = "lmms-lab/VQAv2",
    seed: Optional[int] = None,
):
    """VQAv2 from `lmms-lab/VQAv2`.

    Splits: ``validation``, ``test``, ``testdev``. Note: `lmms-lab/VQAv2`
    does not host the train split — for pretraining you still need the
    original VQAv2 train set from another source.

    Schema: ``image`` (PIL), ``question`` (str), ``question_id`` (int),
    ``multiple_choice_answer`` (str), ``answers`` (list of 10 annotations),
    ``image_id`` (int).
    """
    if split == "val":
        split = "validation"
    if split not in ("validation", "test", "testdev"):
        raise ValueError(
            f"split must be one of validation/test/testdev, got {split!r}"
        )

    ds = load_dataset(hub_id, split=split)
    ds = ds.cast_column("image", DSImage(decode=True))
    ds = ds.with_transform(
        make_vlp_transform(
            image_size,
            image_columns={"image": "image"},
            text_column="question",
            multi_caption=False,
            seed=seed,
        )
    )
    return ds


def load_nlvr2(
    split: str,
    image_size: int,
    hub_id: str = "lmms-lab/NLVR2",
    seed: Optional[int] = None,
):
    """NLVR2 from `lmms-lab/NLVR2`.

    Splits: ``train``, ``dev``, ``test``.

    Schema: ``sentence`` (str), ``left_image`` (PIL), ``right_image`` (PIL),
    ``label`` (str or int).

    Output schema after transform: ``image_0``, ``image_1``, ``text``,
    ``label``. Pair with ``VLPCollator(image_keys=("image_0", "image_1"))``.
    """
    if split == "val":
        split = "dev"
    if split not in ("train", "dev", "test"):
        raise ValueError(f"split must be one of train/dev/test, got {split!r}")

    ds = load_dataset(hub_id, split=split)
    ds = ds.cast_column("left_image", DSImage(decode=True))
    ds = ds.cast_column("right_image", DSImage(decode=True))
    ds = ds.with_transform(
        make_vlp_transform(
            image_size,
            image_columns={"left_image": "image_0", "right_image": "image_1"},
            text_column="sentence",
            multi_caption=False,
            seed=seed,
        )
    )
    return ds


def _load_refcoco_family(
    hub_id: str,
    split: str,
    image_size: int,
    allowed_splits,
    seed: Optional[int] = None,
):
    if split not in allowed_splits:
        raise ValueError(
            f"split must be one of {sorted(allowed_splits)}, got {split!r}"
        )
    ds = load_dataset(hub_id, split=split)
    ds = ds.cast_column("image", DSImage(decode=True))
    ds = ds.with_transform(
        make_vlp_transform(
            image_size,
            image_columns={"image": "image"},
            text_column="question",
            multi_caption=False,
            seed=seed,
        )
    )
    return ds


def load_refcoco(split, image_size, seed=None):
    """RefCOCO from `lmms-lab/RefCOCO`. Splits: val, test, testA, testB.

    Schema: ``image``, ``question`` (referring expression), ``answer`` (list),
    ``bbox`` ([x, y, w, h]), ``segmentation``, ``file_name``.
    """
    return _load_refcoco_family(
        "lmms-lab/RefCOCO", split, image_size,
        allowed_splits={"val", "test", "testA", "testB"},
        seed=seed,
    )


def load_refcocoplus(split, image_size, seed=None):
    """RefCOCO+ from `lmms-lab/RefCOCOplus`. Splits: val, testA, testB."""
    return _load_refcoco_family(
        "lmms-lab/RefCOCOplus", split, image_size,
        allowed_splits={"val", "testA", "testB"},
        seed=seed,
    )


def load_refcocog(split, image_size, seed=None):
    """RefCOCOg from `lmms-lab/RefCOCOg`. Splits: val, test."""
    return _load_refcoco_family(
        "lmms-lab/RefCOCOg", split, image_size,
        allowed_splits={"val", "test"},
        seed=seed,
    )
