"""
Offline tests for the modern `renaissance.data` layer.

Each test builds a tiny in-memory `datasets.Dataset` matching the schema
of the corresponding HF Hub dataset (`nlphuji/flickr30k`, `lmms-lab/VQAv2`,
`lmms-lab/NLVR2`, `lmms-lab/RefCOCO`), runs it through
`make_vlp_transform` + `VLPCollator`, and asserts the batch dict has the
shape the existing encoders expect.

No Hub fetch, no Arrow files, CPU-only.
"""

import io

import pytest
import torch
from datasets import Dataset, Features, IterableDataset
from datasets import Image as DSImage
from datasets import Sequence, Value
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from renaissance.data import (
    VLPCollator,
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
    make_vlp_transform,
)

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


def _img_field(i, offset=0):
    return {"bytes": _png_bytes(((i + offset) * 30 % 255, 100, 200)), "path": None}


def _mlm_collator_or_skip(tokenizer, **kwargs):
    """Local env has TF/numpy mismatch in `transformers.data.data_collator`.
    CI installs no TF and runs fine."""
    try:
        kwargs.setdefault("image_size", IMAGE_SIZE)
        return VLPCollator(tokenizer, max_text_len=TEXT_LEN, do_mlm=True, **kwargs)
    except (ImportError, AttributeError, RuntimeError) as e:
        pytest.skip(f"DataCollatorForLanguageModeling unavailable: {e}")


# ---------------------------------------------------------------------------
# Flickr30k (multi-caption)
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_flickr_ds():
    rows = []
    for i in range(BS * 2):
        rows.append({
            "image": _img_field(i),
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
        make_vlp_transform(text_column="caption", multi_caption=True, seed=0)
    )
    return ds


def test_flickr_transform_yields_pil_image_and_string_text(synthetic_flickr_ds):
    """Dataset-level transform now returns PIL; collator does the
    PIL→Tensor conversion at batch time."""
    row = synthetic_flickr_ds[0]
    assert isinstance(row["image"], Image.Image)
    assert isinstance(row["text"], str)
    assert row["text"].startswith("a synthetic caption number 0.")


def test_flickr_collator_shapes(synthetic_flickr_ds, tokenizer):
    collator = _mlm_collator_or_skip(tokenizer, do_itm=True)
    loader = DataLoader(synthetic_flickr_ds, batch_size=BS, collate_fn=collator)
    batch = next(iter(loader))

    assert batch["image"][0].shape == (BS, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert batch["text_ids"].shape == (BS, TEXT_LEN)
    assert batch["text_ids_mlm"].shape == (BS, TEXT_LEN)
    assert batch["false_image_0"][0].shape == (BS, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert batch["img_id"] == [0, 1, 2, 3]


def test_flickr_collator_passes_through_task_fields(synthetic_flickr_ds, tokenizer):
    collator = VLPCollator(tokenizer, max_text_len=TEXT_LEN, do_mlm=False, do_itm=False, image_size=IMAGE_SIZE)
    loader = DataLoader(synthetic_flickr_ds, batch_size=BS, collate_fn=collator)
    batch = next(iter(loader))

    assert "text_ids_mlm" not in batch
    assert "false_image_0" not in batch
    assert batch["img_id"] == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# VQAv2 (single question, pass-through qid/answers)
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_vqav2_ds():
    rows = []
    for i in range(BS):
        rows.append({
            "image": _img_field(i),
            "question": f"what color is object {i}?",
            "question_id": 1000 + i,
            "multiple_choice_answer": "red",
            "answers": ["red"] * 10,
            "image_id": i,
        })
    features = Features({
        "image": DSImage(),
        "question": Value("string"),
        "question_id": Value("int64"),
        "multiple_choice_answer": Value("string"),
        "answers": Sequence(Value("string")),
        "image_id": Value("int64"),
    })
    ds = Dataset.from_list(rows, features=features)
    ds = ds.with_transform(
        make_vlp_transform(text_column="question", multi_caption=False)
    )
    return ds


def test_vqav2_pipeline(synthetic_vqav2_ds, tokenizer):
    collator = VLPCollator(tokenizer, max_text_len=TEXT_LEN, do_mlm=False, do_itm=False, image_size=IMAGE_SIZE)
    loader = DataLoader(synthetic_vqav2_ds, batch_size=BS, collate_fn=collator)
    batch = next(iter(loader))

    assert batch["image"][0].shape == (BS, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert batch["text_ids"].shape == (BS, TEXT_LEN)
    assert batch["question_id"] == [1000, 1001, 1002, 1003]
    assert batch["multiple_choice_answer"] == ["red"] * BS
    assert len(batch["answers"]) == BS


# ---------------------------------------------------------------------------
# NLVR2 (two images, image_0 / image_1)
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_nlvr2_ds():
    rows = []
    for i in range(BS):
        rows.append({
            "left_image": _img_field(i, offset=0),
            "right_image": _img_field(i, offset=100),
            "sentence": f"the left image shows {i} dogs.",
            "label": "True" if i % 2 == 0 else "False",
        })
    features = Features({
        "left_image": DSImage(),
        "right_image": DSImage(),
        "sentence": Value("string"),
        "label": Value("string"),
    })
    ds = Dataset.from_list(rows, features=features)
    ds = ds.with_transform(
        make_vlp_transform(
            image_columns={"left_image": "image_0", "right_image": "image_1"},
            text_column="sentence",
            multi_caption=False,
        )
    )
    return ds


def test_nlvr2_pipeline(synthetic_nlvr2_ds, tokenizer):
    collator = VLPCollator(
        tokenizer, max_text_len=TEXT_LEN, do_mlm=False, do_itm=False,
        image_keys=("image_0", "image_1"), image_size=IMAGE_SIZE,
    )
    loader = DataLoader(synthetic_nlvr2_ds, batch_size=BS, collate_fn=collator)
    batch = next(iter(loader))

    assert batch["image_0"][0].shape == (BS, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert batch["image_1"][0].shape == (BS, 3, IMAGE_SIZE, IMAGE_SIZE)
    # NLVR2 left/right are different colors — the tensors should differ.
    assert not torch.allclose(batch["image_0"][0], batch["image_1"][0])
    assert batch["text_ids"].shape == (BS, TEXT_LEN)
    assert batch["label"] == ["True", "False", "True", "False"]
    assert "image" not in batch  # collator should not synthesize a default "image" key


# ---------------------------------------------------------------------------
# RefCOCO (bbox pass-through)
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_refcoco_ds():
    rows = []
    for i in range(BS):
        rows.append({
            "image": _img_field(i),
            "question": f"the {['red','blue','green','yellow'][i]} thing",
            "answer": [f"the {['red','blue','green','yellow'][i]} thing"],
            "bbox": [float(i), float(i + 1), 10.0, 20.0],
            "file_name": f"COCO_{i:012d}.jpg",
        })
    features = Features({
        "image": DSImage(),
        "question": Value("string"),
        "answer": Sequence(Value("string")),
        "bbox": Sequence(Value("float32"), length=4),
        "file_name": Value("string"),
    })
    ds = Dataset.from_list(rows, features=features)
    ds = ds.with_transform(
        make_vlp_transform(text_column="question", multi_caption=False)
    )
    return ds


def test_refcoco_pipeline(synthetic_refcoco_ds, tokenizer):
    collator = VLPCollator(tokenizer, max_text_len=TEXT_LEN, do_mlm=False, do_itm=False, image_size=IMAGE_SIZE)
    loader = DataLoader(synthetic_refcoco_ds, batch_size=BS, collate_fn=collator)
    batch = next(iter(loader))

    assert batch["image"][0].shape == (BS, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert batch["text_ids"].shape == (BS, TEXT_LEN)
    # Bbox passed through as a list of length-4 sequences.
    assert len(batch["bbox"]) == BS
    assert all(len(bb) == 4 for bb in batch["bbox"])
    assert batch["file_name"][0].startswith("COCO_")


# ---------------------------------------------------------------------------
# COCO Karpathy (multi-caption, embedded images)
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_coco_ds():
    """Schema matches `namkha1032/coco-karpathy`: image_id, image, captions
    (list[str] of length 5-7), plus the awkwardly-named width column the
    loader drops on the real Hub."""
    rows = []
    for i in range(BS * 2):
        rows.append({
            "image_id": str(100000 + i),
            "image": _img_field(i),
            "captions": [f"a coco caption {i}.{j}" for j in range(5)],
        })
    features = Features({
        "image_id": Value("string"),
        "image": DSImage(),
        "captions": Sequence(Value("string")),
    })
    ds = Dataset.from_list(rows, features=features)
    ds = ds.with_transform(
        make_vlp_transform(text_column="captions", multi_caption=True, seed=0)
    )
    return ds


def test_coco_pipeline(synthetic_coco_ds, tokenizer):
    collator = VLPCollator(tokenizer, max_text_len=TEXT_LEN, do_mlm=False, do_itm=False, image_size=IMAGE_SIZE)
    loader = DataLoader(synthetic_coco_ds, batch_size=BS, collate_fn=collator)
    batch = next(iter(loader))

    assert batch["image"][0].shape == (BS, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert batch["text_ids"].shape == (BS, TEXT_LEN)
    assert len(batch["text"]) == BS
    assert batch["text"][0].startswith("a coco caption 0.")
    assert batch["image_id"] == ["100000", "100001", "100002", "100003"]


# ---------------------------------------------------------------------------
# CC3M / CC12M (WebDataset, streaming)
# ---------------------------------------------------------------------------

def _wds_rows(n, prefix="cc3m"):
    """Mimic a pixparse/cc3m-wds row: {__key__, jpg (bytes), txt}."""
    return [
        {
            "__key__": f"{prefix}/shard0/{i:09d}",
            "jpg": _png_bytes((i * 40 % 255, 50, 150)),
            "txt": f"a {prefix}-style caption number {i}",
        }
        for i in range(n)
    ]


def test_cc3m_streaming_pipeline(tokenizer):
    # Build an IterableDataset to match what load_cc3m(streaming=True) returns.
    # `to_iterable_dataset()` sidesteps `from_generator`, which would otherwise
    # break in environments with a dill version that's out of sync with HF
    # datasets.
    features = Features({
        "__key__": Value("string"),
        "jpg": Value("binary"),
        "txt": Value("string"),
    })
    from renaissance.data.loaders import _WDS_OUT_FEATURES

    base = Dataset.from_list(_wds_rows(BS * 2), features=features)
    ds = base.to_iterable_dataset()

    transform = make_vlp_transform(
        image_columns={"jpg": "image"},
        text_column="txt",
        multi_caption=False,
    )
    # Mirrors load_cc3m: drop all source columns from the output, declare
    # explicit features so downstream feature encoding doesn't trip.
    ds = ds.map(
        transform, batched=True,
        remove_columns=["__key__", "jpg", "txt"],
        features=_WDS_OUT_FEATURES,
    )

    collator = VLPCollator(tokenizer, max_text_len=TEXT_LEN, do_mlm=False, do_itm=False, image_size=IMAGE_SIZE)
    loader = DataLoader(ds, batch_size=BS, collate_fn=collator)
    batch = next(iter(loader))

    assert batch["image"][0].shape == (BS, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert batch["text_ids"].shape == (BS, TEXT_LEN)
    assert len(batch["text"]) == BS
    assert batch["text"][0].startswith("a cc3m-style caption number")
    # Source columns dropped; only image+text remain.
    assert "__key__" not in batch
    assert "jpg" not in batch
    assert "txt" not in batch


def test_raw_bytes_image_handled():
    """_to_pil should accept raw bytes (WebDataset image format) directly."""
    from renaissance.data.transforms import _to_pil

    img = _to_pil(_png_bytes((10, 20, 30)))
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"


# ---------------------------------------------------------------------------
# Visual Genome (region descriptions → multi-caption)
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_vg_ds():
    """Mirror of ranjaykrishna/visual_genome (region_descriptions config).
    Each row has an `image` and a `regions` list-of-structs containing
    `phrase` and bbox info. We replicate the .map(extract_phrases) step
    inline so the test focuses on the transform + collator part of the
    pipeline."""
    rows = []
    for i in range(BS):
        rows.append({
            "image": _img_field(i),
            "image_id": 1000 + i,
            "regions": [
                {
                    "region_id": 10 * i + j,
                    "image_id": 1000 + i,
                    "phrase": f"a vg phrase {i}.{j}",
                    "x": j,
                    "y": j,
                    "width": 10,
                    "height": 10,
                }
                for j in range(4)
            ],
        })
    # `[{...}]` (list-of-struct) rather than `Sequence({...})` — the latter
    # is the struct-of-arrays form and `Dataset.from_list` with list-of-dict
    # rows doesn't round-trip cleanly through it on newer `datasets`.
    features = Features({
        "image": DSImage(),
        "image_id": Value("int64"),
        "regions": [{
            "region_id": Value("int64"),
            "image_id": Value("int64"),
            "phrase": Value("string"),
            "x": Value("int64"),
            "y": Value("int64"),
            "width": Value("int64"),
            "height": Value("int64"),
        }],
    })
    ds = Dataset.from_list(rows, features=features)
    # Same logic the loader runs.
    def _extract_phrases(batch):
        captions = []
        for regions in batch["regions"]:
            if isinstance(regions, dict):
                captions.append(list(regions["phrase"]))
            else:
                captions.append([r["phrase"] for r in regions])
        return {"caption": captions}
    ds = ds.map(_extract_phrases, batched=True, remove_columns=["regions"])
    ds = ds.with_transform(
        make_vlp_transform(text_column="caption", multi_caption=True, seed=0)
    )
    return ds


def test_vg_pipeline(synthetic_vg_ds, tokenizer):
    collator = VLPCollator(tokenizer, max_text_len=TEXT_LEN, do_mlm=False, do_itm=False, image_size=IMAGE_SIZE)
    loader = DataLoader(synthetic_vg_ds, batch_size=BS, collate_fn=collator)
    batch = next(iter(loader))

    assert batch["image"][0].shape == (BS, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert batch["text_ids"].shape == (BS, TEXT_LEN)
    assert batch["text"][0].startswith("a vg phrase 0.")
    assert batch["image_id"] == [1000, 1001, 1002, 1003]


# ---------------------------------------------------------------------------
# SNLI-VE (image-premise + hypothesis text + label)
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_snli_ve_ds():
    rows = []
    labels = ["entailment", "neutral", "contradiction", "entailment"]
    for i in range(BS):
        rows.append({
            "image": _img_field(i),
            "filename": f"{2000 + i}.jpg",
            "premise": f"premise about image {i}",
            "hypothesis": f"hypothesis number {i}",
            "label": labels[i],
        })
    features = Features({
        "image": DSImage(),
        "filename": Value("string"),
        "premise": Value("string"),
        "hypothesis": Value("string"),
        "label": Value("string"),
    })
    ds = Dataset.from_list(rows, features=features)
    ds = ds.with_transform(
        make_vlp_transform(text_column="hypothesis", multi_caption=False)
    )
    return ds


def test_snli_ve_pipeline(synthetic_snli_ve_ds, tokenizer):
    collator = VLPCollator(tokenizer, max_text_len=TEXT_LEN, do_mlm=False, do_itm=False, image_size=IMAGE_SIZE)
    loader = DataLoader(synthetic_snli_ve_ds, batch_size=BS, collate_fn=collator)
    batch = next(iter(loader))

    assert batch["image"][0].shape == (BS, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert batch["text_ids"].shape == (BS, TEXT_LEN)
    assert batch["text"][0].startswith("hypothesis number")
    assert batch["label"][0] == "entailment"
    # `premise` passed through but unused as input text.
    assert "premise" in batch and len(batch["premise"]) == BS


# ---------------------------------------------------------------------------
# GLUE (text-only) — exercises image_keys=() and text_pair handling
# ---------------------------------------------------------------------------

def test_glue_paired_pipeline(tokenizer):
    """MRPC-style: text + text_pair → tokenizer pair encoding."""
    rows = [
        {"text": "the dog runs.", "text_pair": "a dog is running.", "label": 1, "idx": 0},
        {"text": "cats sleep.", "text_pair": "the cat is asleep.", "label": 1, "idx": 1},
        {"text": "the sky is blue.", "text_pair": "blue is the sky.", "label": 1, "idx": 2},
        {"text": "i ate an apple.", "text_pair": "a banana was eaten.", "label": 0, "idx": 3},
    ]
    ds = Dataset.from_list(rows)

    collator = VLPCollator(
        tokenizer, max_text_len=TEXT_LEN, do_mlm=False, do_itm=False,
        image_keys=(),
    )
    loader = DataLoader(ds, batch_size=BS, collate_fn=collator)
    batch = next(iter(loader))

    assert "image" not in batch
    assert "false_image_0" not in batch
    assert batch["text_ids"].shape == (BS, TEXT_LEN)
    assert batch["text_masks"].shape == (BS, TEXT_LEN)
    assert batch["text_pair"][0] == "a dog is running."
    assert batch["label"] == [1, 1, 1, 0]
    assert batch["idx"] == [0, 1, 2, 3]


def test_glue_single_sentence_pipeline(tokenizer):
    """SST2/CoLA-style: single text, no text_pair."""
    rows = [
        {"text": "this movie is great.", "label": 1, "idx": 0},
        {"text": "the food was terrible.", "label": 0, "idx": 1},
        {"text": "an amazing performance.", "label": 1, "idx": 2},
        {"text": "boring and slow.", "label": 0, "idx": 3},
    ]
    ds = Dataset.from_list(rows)

    collator = VLPCollator(
        tokenizer, max_text_len=TEXT_LEN, do_mlm=False, do_itm=False,
        image_keys=(),
    )
    loader = DataLoader(ds, batch_size=BS, collate_fn=collator)
    batch = next(iter(loader))

    assert "text_pair" not in batch
    assert batch["text_ids"].shape == (BS, TEXT_LEN)
    assert batch["label"] == [1, 0, 1, 0]


def test_text_only_collator_ignores_itm_flag(tokenizer):
    """do_itm=True with image_keys=() should silently no-op rather than crash."""
    rows = [{"text": "hello world.", "label": 0}] * BS
    ds = Dataset.from_list(rows)
    collator = VLPCollator(
        tokenizer, max_text_len=TEXT_LEN, do_mlm=False, do_itm=True,
        image_keys=(),
    )
    loader = DataLoader(ds, batch_size=BS, collate_fn=collator)
    batch = next(iter(loader))
    assert "false_image_0" not in batch


# ---------------------------------------------------------------------------
# Loader signature smoke checks (don't actually hit the Hub)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "loader,bad_split",
    [
        (load_flickr30k, "bogus"),
        (load_vqav2, "bogus"),
        (load_nlvr2, "bogus"),
        (load_refcoco, "bogus"),
        (load_refcocoplus, "bogus"),
        (load_refcocog, "bogus"),
        (load_cc3m, "bogus"),
        (load_cc12m, "bogus"),
        (load_coco_karpathy, "bogus"),
        (load_visual_genome, "bogus"),
        (load_snli_ve, "bogus"),
    ],
)
def test_loader_rejects_bad_split(loader, bad_split):
    with pytest.raises(ValueError, match="split must be one of"):
        loader(split=bad_split)


def test_load_sbu_rejects_empty_path(tmp_path):
    """SBU has no Hub mirror; require local img2dataset output and fail
    loudly when the directory is empty."""
    with pytest.raises(FileNotFoundError, match="No .tar WebDataset shards"):
        load_sbu(path=str(tmp_path))


def test_load_glue_rejects_bad_task():
    with pytest.raises(ValueError, match="task must be one of"):
        load_glue(task="bogus", split="train")


def test_load_glue_rejects_bad_split():
    with pytest.raises(ValueError, match="split must be one of"):
        load_glue(task="mrpc", split="bogus")
