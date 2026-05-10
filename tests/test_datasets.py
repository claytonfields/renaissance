"""
Integration tests for the HF-Datasets-backed data layer (Step 2).

Builds tiny in-memory HF Datasets that match the column schema of each
dataset class, then verifies that __getitem__ and collate produce the
batch shapes the model objectives expect.  No Arrow files or external
downloads needed.
"""

import io
import pytest
import torch
import numpy as np
from PIL import Image
from datasets import Dataset
from transformers import AutoTokenizer, AutoFeatureExtractor, DataCollatorForWholeWordMask
from renaissance.transforms import keys_to_transforms

from tests.conftest import BS, TEXT_LEN

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMG_SIZE = 32
N = 6  # samples — enough for draw_false_image random sampling
TEXT_ENC = "google/electra-small-discriminator"
IMG_ENC  = "facebook/deit-tiny-patch16-224"


# ---------------------------------------------------------------------------
# Data factories
# ---------------------------------------------------------------------------

def _img_bytes(h=IMG_SIZE, w=IMG_SIZE):
    arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


def _caption_table(n=N):
    """Schema: image (bytes), caption (list[str])."""
    return Dataset.from_dict({
        "image":   [_img_bytes() for _ in range(n)],
        "caption": [["a cat", "a dog"] for _ in range(n)],
    })


def _snli_table(n=N):
    """Schema: image (bytes), sentences (list[list[str,str]]), labels (list[int]).
    sentences[i] is a list of [sentence1, sentence2] pairs for image i.
    t[1] is the hypothesis (what base_dataset extracts).
    """
    return Dataset.from_dict({
        "image":     [_img_bytes() for _ in range(n)],
        "sentences": [
            [["a cat sits on a mat.", "a cat is present."],
             ["a cat sits.", "no animal is visible."]]
            for _ in range(n)
        ],
        "labels": [[2, 0] for _ in range(n)],
    })


def _nlvr2_table(n=N):
    """Schema: image_0, image_1 (bytes), questions (list[str]), answers (list[str])."""
    return Dataset.from_dict({
        "image_0":   [_img_bytes() for _ in range(n)],
        "image_1":   [_img_bytes() for _ in range(n)],
        "questions": [["is there a cat?"] for _ in range(n)],
        "answers":   [["True"] for _ in range(n)],
    })


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained(TEXT_ENC)
    tok.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    return tok


@pytest.fixture(scope="module")
def processor():
    return AutoFeatureExtractor.from_pretrained(IMG_ENC)


@pytest.fixture(scope="module")
def mlm_collator(tokenizer):
    return DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm_probability=0.15)


# ---------------------------------------------------------------------------
# Helper: build a dataset instance without file I/O
# ---------------------------------------------------------------------------

def _build_ds(cls, table, text_column, remove_duplicate=True, **extra):
    """
    Instantiate cls without calling __init__ (which would try to open .arrow
    files), then set all attributes BaseDataset.__init__ would have set from
    the loaded table.
    """
    ds = cls.__new__(cls)
    ds.split = extra.pop("split", "train")
    ds.names = extra.pop("names", ["test"])
    ds.image_only = False
    ds.clip_transform = False
    ds.max_text_len = TEXT_LEN
    ds.image_size = IMG_SIZE
    ds.draw_false_image = extra.pop("draw_false_image", 1)
    ds.draw_false_text = 0
    ds.hugging_face = False
    for k, v in extra.items():
        setattr(ds, k, v)

    ds.transforms = keys_to_transforms(["imagenet"], size=IMG_SIZE)
    ds.table = table
    ds.text_column_name = text_column
    ds.table_names = ["test_table"] * len(table)

    raw = table[text_column]
    if text_column == "sentences":
        # snli: each row is list of [s1, s2] pairs; extract s2 (hypothesis)
        all_texts = [[pair[1].strip() for pair in row] for row in raw]
    elif remove_duplicate:
        all_texts = [list(set(row)) for row in raw]
    else:
        all_texts = [list(row) for row in raw]

    ds.all_texts = all_texts
    ds.index_mapper = {}
    j = 0
    for i, texts in enumerate(all_texts):
        for _j in range(len(texts)):
            ds.index_mapper[j] = (i, _j)
            j += 1

    return ds


# ---------------------------------------------------------------------------
# Core: Dataset wrapping a PyArrow table
# ---------------------------------------------------------------------------

class TestHFDatasetWrapping:
    """Verify that Dataset(pa_table) provides the same access patterns as
    the old table[col][idx].as_py() PyArrow API."""

    def test_row_access_bytes(self):
        import pyarrow as pa
        original_bytes = _img_bytes()
        pa_table = pa.table({"image": [original_bytes], "caption": [["hello"]]})
        ds = Dataset(pa_table)
        assert ds[0]["image"] == original_bytes

    def test_column_access_list(self):
        import pyarrow as pa
        pa_table = pa.table({"caption": [["a", "b"], ["c"]]})
        ds = Dataset(pa_table)
        assert ds["caption"] == [["a", "b"], ["c"]]

    def test_nested_label_access(self):
        import pyarrow as pa
        pa_table = pa.table({"labels": [[0, 2], [1, 0]]})
        ds = Dataset(pa_table)
        assert ds[1]["labels"][0] == 1

    def test_concatenate_datasets(self):
        from datasets import concatenate_datasets
        import pyarrow as pa
        t1 = Dataset(pa.table({"x": [1, 2]}))
        t2 = Dataset(pa.table({"x": [3, 4]}))
        combined = concatenate_datasets([t1, t2])
        assert len(combined) == 4
        assert combined["x"] == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Caption dataset (coco / vg / f30k / sbu schema)
# ---------------------------------------------------------------------------

class TestCaptionDataset:

    @pytest.fixture(scope="class")
    def ds(self, tokenizer, processor):
        from renaissance.datasets.coco_caption_karpathy_dataset import CocoCaptionKarpathyDataset
        return _build_ds(
            CocoCaptionKarpathyDataset, _caption_table(), "caption",
            tokenizer=tokenizer, processor=processor,
        )

    def test_getitem_keys(self, ds):
        item = ds[0]
        assert "image" in item
        assert "text" in item
        assert "false_image_0" in item

    def test_image_shape(self, ds):
        item = ds[0]
        assert item["image"][0].shape == (3, IMG_SIZE, IMG_SIZE)

    def test_collate_shapes(self, ds, mlm_collator):
        batch = [ds[i] for i in range(BS)]
        out = ds.collate(batch, mlm_collator)
        assert out["image"][0].shape == (BS, 3, IMG_SIZE, IMG_SIZE)
        assert out["text_ids"].shape == (BS, TEXT_LEN)
        assert out["text_ids_mlm"].shape == (BS, TEXT_LEN)
        assert out["text_labels_mlm"].shape == (BS, TEXT_LEN)
        assert out["text_masks"].shape == (BS, TEXT_LEN)


# ---------------------------------------------------------------------------
# SNLI dataset
# ---------------------------------------------------------------------------

class TestSNLIDataset:

    @pytest.fixture(scope="class")
    def ds(self, tokenizer, processor):
        from renaissance.datasets.snli_dataset import SNLIDataset
        return _build_ds(
            SNLIDataset, _snli_table(), "sentences", remove_duplicate=False,
            tokenizer=tokenizer, processor=processor,
        )

    def test_getitem_label(self, ds):
        item = ds[0]
        assert "labels" in item
        assert isinstance(item["labels"], int)

    def test_collate_shapes(self, ds, mlm_collator):
        batch = [ds[i] for i in range(BS)]
        out = ds.collate(batch, mlm_collator)
        assert out["image"][0].shape[0] == BS
        assert "labels" in out


# ---------------------------------------------------------------------------
# NLVR2 dataset
# ---------------------------------------------------------------------------

class TestNLVR2Dataset:

    @pytest.fixture(scope="class")
    def ds(self, tokenizer, processor):
        from renaissance.datasets.nlvr2_dataset import NLVR2Dataset
        ds = _build_ds(
            NLVR2Dataset, _nlvr2_table(), "questions", remove_duplicate=False,
            draw_false_image=0, tokenizer=tokenizer, processor=processor,
        )
        return ds

    def test_getitem_dual_images(self, ds):
        item = ds[0]
        assert "image_0" in item
        assert "image_1" in item

    def test_answer_is_bool(self, ds):
        item = ds[0]
        assert isinstance(item["answers"], bool)

    def test_collate_shapes(self, ds, mlm_collator):
        batch = [ds[i] for i in range(BS)]
        out = ds.collate(batch, mlm_collator)
        assert out["image_0"][0].shape[0] == BS
        assert out["image_1"][0].shape[0] == BS
