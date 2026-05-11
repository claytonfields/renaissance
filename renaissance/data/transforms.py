"""
Per-item transforms applied via `Dataset.with_transform` (single-dataset
path) or `.map` (multi-dataset interleave path).

The transform produces PIL images and selected captions only — final
image tensorization happens in `VLPCollator.__call__` so the dataset
schema stays Arrow-compatible. This matches modern HF/PyTorch practice
and unblocks `interleave_datasets`, which can't deal with Tensor columns
during feature inference.

- `make_image_transform`: torchvision pipeline (resize → ToTensor →
  normalize). Used by the collator at batch time, not by datasets.
- `make_vlp_transform`: the callable handed to `with_transform`. Picks
  one caption per row for multi-caption datasets and renames image/text
  columns. Image values stay PIL.
"""

import random
from typing import Mapping, Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def make_image_transform(image_size, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def _to_pil(img):
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, np.ndarray):
        return Image.fromarray(img).convert("RGB")
    if isinstance(img, torch.Tensor):
        return T.ToPILImage()(img).convert("RGB")
    if isinstance(img, (bytes, bytearray)):
        from io import BytesIO
        return Image.open(BytesIO(img)).convert("RGB")
    if isinstance(img, dict) and "bytes" in img:
        from io import BytesIO
        return Image.open(BytesIO(img["bytes"])).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(img)}")


def make_vlp_transform(
    image_columns: Optional[Mapping[str, str]] = None,
    text_column: str = "caption",
    text_output: str = "text",
    multi_caption: bool = True,
    seed: Optional[int] = None,
    pass_through: bool = True,
    bbox_column: Optional[str] = None,
    bbox_output: str = "bbox",
):
    """Return a `with_transform` callable that processes a batch dict.

    Output image columns hold PIL images, not tensors — the collator does
    the final tensor conversion. This keeps the dataset schema
    Arrow-compatible, which is required for `interleave_datasets`.

    Parameters
    ----------
    image_columns
        Mapping of source column name → output column name. Defaults to
        ``{"image": "image"}``. NLVR2 uses
        ``{"left_image": "image_0", "right_image": "image_1"}``.
    text_column
        Column holding the text payload. For caption datasets a list of
        strings; for VQA/NLVR2/RefCOCO a single string.
    text_output
        Output key for the chosen string. Default ``"text"``.
    multi_caption
        If True, randomly pick one string from a list column. If False,
        ``text_column`` is assumed to hold a single string per row.
    seed
        Optional seed for the caption-picking RNG.
    pass_through
        If True (default), pass through any non-image, non-text columns
        unchanged. Set False for the interleave path where downstream code
        wants only `{image, text}`.
    bbox_column
        If set, the source column holding a COCO-format ``[x, y, w, h]``
        bbox in original-image pixel coordinates. The transform reads
        ``(W, H)`` from the first image column and emits a normalized
        ``[x1, y1, x2, y2]`` bbox in ``[0, 1]`` at ``bbox_output``. Scale-
        invariant against the collator's image resize. The source column
        is dropped from ``pass_through``.
    bbox_output
        Output key for the rescaled bbox. Default ``"bbox"``.
    """
    if image_columns is None:
        image_columns = {"image": "image"}

    rng = random.Random(seed)

    def transform(batch):
        out = {}

        pil_by_dst = {}
        for src, dst in image_columns.items():
            pils = [_to_pil(img) for img in batch[src]]
            pil_by_dst[dst] = pils
            out[dst] = pils

        if multi_caption:
            out[text_output] = [rng.choice(caps) for caps in batch[text_column]]
        else:
            out[text_output] = [
                t if isinstance(t, str) else t[0] for t in batch[text_column]
            ]

        if bbox_column is not None:
            ref_dst = next(iter(image_columns.values()))
            ref_pils = pil_by_dst[ref_dst]
            out[bbox_output] = [
                _normalize_bbox_xywh_to_xyxy(b, pil.size)
                for b, pil in zip(batch[bbox_column], ref_pils)
            ]

        if pass_through:
            skip_keys = set(image_columns.keys()) | {text_column}
            if bbox_column is not None:
                skip_keys.add(bbox_column)
            for k, v in batch.items():
                if k in skip_keys:
                    continue
                out[k] = v
        return out

    return transform


def _normalize_bbox_xywh_to_xyxy(bbox, image_size):
    """Convert ``[x, y, w, h]`` pixel-coord bbox to normalized ``[x1, y1, x2, y2]``.

    ``image_size`` is the PIL ``(width, height)`` tuple.
    """
    x, y, w, h = (float(v) for v in bbox)
    W, H = image_size
    return [x / W, y / H, (x + w) / W, (y + h) / H]
