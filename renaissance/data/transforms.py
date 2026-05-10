"""
Per-item transforms applied via `Dataset.with_transform`.

Two pieces:
- `make_image_transform`: torchvision pipeline (resize → ToTensor → normalize).
- `make_vlp_transform`: the callable handed to `with_transform`. Converts the
  image column to a tensor and, for multi-caption datasets, picks one caption
  per row. Tokenization happens later in `VLPCollator` so changing tokenizers
  doesn't invalidate a `.map` cache.
"""

import random

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
    if isinstance(img, dict) and "bytes" in img:
        from io import BytesIO
        return Image.open(BytesIO(img["bytes"])).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(img)}")


def make_vlp_transform(
    image_size,
    image_column="image",
    caption_column="caption",
    multi_caption=True,
    seed=None,
):
    """Return a `with_transform` callable that processes a batch dict.

    The transform converts each image to a normalized tensor and reduces the
    caption column to a single string per row (random pick when
    `multi_caption=True`). Other columns are passed through unchanged.
    """
    image_tf = make_image_transform(image_size)
    rng = random.Random(seed)

    def transform(batch):
        out = {"image": [], "text": []}
        images = batch[image_column]
        captions = batch[caption_column]
        for img, caps in zip(images, captions):
            out["image"].append(image_tf(_to_pil(img)))
            if multi_caption:
                out["text"].append(rng.choice(caps))
            else:
                out["text"].append(caps if isinstance(caps, str) else caps[0])

        for k, v in batch.items():
            if k in (image_column, caption_column):
                continue
            out[k] = v
        return out

    return transform
