"""
Offline tests for the modern `renaissance.data` layer.

Builds a tiny in-memory `datasets.Dataset` matching the `nlphuji/flickr30k`
schema, runs it through `make_vlp_transform` + `VLPCollator`, and asserts
the batch dict has the shape the existing encoders expect.

No Hub fetch, no Arrow files, CPU-only.
"""

import io

import pytest
import torch
from datasets import Dataset, Features
from datasets import Image as DSImage
from datasets import Sequence, Value
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from renaissance.data import VLPCollator, load_flickr30k, make_vlp_transform

TEXT_ENC = "google/electra-small-discriminator"
IMAGE_SIZE = 32
TEXT_LEN = 16
BS = 4


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(TEXT_ENC)


def _png_bytes(color):
    buf = io.BytesIO()
    Image.new("RGB", (40, 60), color=color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def synthetic_flickr_ds():
    """In-memory Dataset matching nlphuji/flickr30k columns."""
    rows = []
    for i in range(BS * 2):
        rows.append({
            "image": {"bytes": _png_bytes((i * 30 % 255, 100, 200)), "path": None},
            "caption": [f"a synthetic caption number {i}.{j}" for j in range(5)],
            "sentids": [10 * i + j for j in range(5)],
            "img_id": i,
            "filename": f"synth_{i}.png",
            "split": "train",
        })
    features = Features({
        "image": DSImage(),
        "caption": Sequence(Value("string")),
        "sentids": Sequence(Value("int64")),
        "img_id": Value("int64"),
        "filename": Value("string"),
        "split": Value("string"),
    })
    ds = Dataset.from_list(rows, features=features)
    ds = ds.with_transform(
        make_vlp_transform(IMAGE_SIZE, "image", "caption", multi_caption=True, seed=0)
    )
    return ds


def test_transform_yields_tensor_image_and_string_text(synthetic_flickr_ds):
    row = synthetic_flickr_ds[0]
    assert isinstance(row["image"], torch.Tensor)
    assert row["image"].shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert isinstance(row["text"], str)
    assert row["text"].startswith("a synthetic caption number 0.")


def _mlm_collator_or_skip(tokenizer):
    """Local env has TF/numpy mismatch in `transformers.data.data_collator`.
    CI installs no TF and runs fine."""
    try:
        return VLPCollator(tokenizer, max_text_len=TEXT_LEN, do_mlm=True, do_itm=True)
    except (ImportError, AttributeError, RuntimeError) as e:
        pytest.skip(f"DataCollatorForLanguageModeling unavailable: {e}")


def test_collator_shapes(synthetic_flickr_ds, tokenizer):
    collator = _mlm_collator_or_skip(tokenizer)
    loader = DataLoader(synthetic_flickr_ds, batch_size=BS, collate_fn=collator)
    batch = next(iter(loader))

    assert isinstance(batch["image"], list) and len(batch["image"]) == 1
    assert batch["image"][0].shape == (BS, 3, IMAGE_SIZE, IMAGE_SIZE)

    assert isinstance(batch["text"], list) and len(batch["text"]) == BS
    assert batch["text_ids"].shape == (BS, TEXT_LEN)
    assert batch["text_masks"].shape == (BS, TEXT_LEN)
    assert batch["text_labels"].shape == (BS, TEXT_LEN)
    assert (batch["text_labels"] == -100).all()

    assert batch["text_ids_mlm"].shape == (BS, TEXT_LEN)
    assert batch["text_labels_mlm"].shape == (BS, TEXT_LEN)

    assert isinstance(batch["false_image_0"], list) and len(batch["false_image_0"]) == 1
    assert batch["false_image_0"][0].shape == (BS, 3, IMAGE_SIZE, IMAGE_SIZE)


def test_collator_passes_through_task_fields(synthetic_flickr_ds, tokenizer):
    collator = VLPCollator(tokenizer, max_text_len=TEXT_LEN, do_mlm=False, do_itm=False)
    loader = DataLoader(synthetic_flickr_ds, batch_size=BS, collate_fn=collator)
    batch = next(iter(loader))

    assert "text_ids_mlm" not in batch
    assert "false_image_0" not in batch
    assert "img_id" in batch and len(batch["img_id"]) == BS
    assert "filename" in batch and len(batch["filename"]) == BS


def test_load_flickr30k_signature():
    """Smoke check that the loader function exists and validates its args
    without actually hitting the Hub (which CI is offline for)."""
    with pytest.raises(ValueError, match="split must be one of"):
        load_flickr30k(split="bogus", image_size=IMAGE_SIZE)
