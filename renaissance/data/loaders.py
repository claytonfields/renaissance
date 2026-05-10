"""
Per-dataset loaders. Each function returns a `datasets.Dataset` (or
`IterableDataset` for streaming sources) with `with_transform` applied so
each row yields `{image: Tensor[3, H, W], text: str, ...}`.

The DataLoader is constructed elsewhere; pair these with `VLPCollator`.
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
    """Load Karpathy-split Flickr30k from the Hub.

    `nlphuji/flickr30k` packages all 31,014 images in a single "test" split
    with a `split` column tagging Karpathy assignment ("train" / "val" /
    "test"). Each row has a `caption` column with 5 strings.
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be one of train/val/test, got {split!r}")

    ds = load_dataset(hub_id, split="test")
    ds = ds.filter(lambda x: x["split"] == split)
    ds = ds.cast_column("image", DSImage(decode=True))
    ds = ds.with_transform(
        make_vlp_transform(
            image_size,
            image_column="image",
            caption_column="caption",
            multi_caption=True,
            seed=seed,
        )
    )
    return ds
