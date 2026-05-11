"""
Per-item transforms applied via `Dataset.with_transform`.

Two pieces:
- `make_image_transform`: torchvision pipeline (resize → ToTensor → normalize).
- `make_vlp_transform`: the callable handed to `with_transform`. Converts one
  or more image columns to tensors and reduces the text column to a single
  string per row (random caption pick when `multi_caption=True`).
  Tokenization happens later in `VLPCollator` so changing tokenizers doesn't
  invalidate any `.map` cache.
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
    image_size: int,
    image_columns: Optional[Mapping[str, str]] = None,
    text_column: str = "caption",
    text_output: str = "text",
    multi_caption: bool = True,
    seed: Optional[int] = None,
):
    """Return a `with_transform` callable that processes a batch dict.

    Parameters
    ----------
    image_size
        Resize target (square).
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
    """
    if image_columns is None:
        image_columns = {"image": "image"}

    image_tf = make_image_transform(image_size)
    rng = random.Random(seed)

    def transform(batch):
        out = {}

        for src, dst in image_columns.items():
            out[dst] = [image_tf(_to_pil(img)) for img in batch[src]]

        if multi_caption:
            out[text_output] = [rng.choice(caps) for caps in batch[text_column]]
        else:
            out[text_output] = [
                t if isinstance(t, str) else t[0] for t in batch[text_column]
            ]

        skip_keys = set(image_columns.keys()) | {text_column}
        for k, v in batch.items():
            if k in skip_keys:
                continue
            out[k] = v
        return out

    return transform
