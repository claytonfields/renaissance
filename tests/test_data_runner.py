"""
Tests for the modern data-layer runner (`renaissance.data.runner`).

Focuses on dispatch and wiring logic — exercises the path from a config
dict to a fully-functional DataLoader without hitting the HF Hub. The
actual `load_*` calls are monkeypatched to return synthetic datasets.
"""

import io

import pytest
import torch
from datasets import Dataset, Features
from datasets import Image as DSImage
from datasets import Sequence, Value
from PIL import Image
from transformers import AutoTokenizer

from renaissance.data import build_collator, build_dataloader, build_dataset
from renaissance.data.runner import _REGISTRY, _kwargs_for, _tokenizer_name

TEXT_ENC = "google/electra-small-discriminator"
IMAGE_SIZE = 32
TEXT_LEN = 16
BS = 4


def _png_bytes(color):
    buf = io.BytesIO()
    Image.new("RGB", (40, 60), color=color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(TEXT_ENC)


def _base_config(**overrides):
    cfg = {
        "datasets": ["coco"],
        "image_size": IMAGE_SIZE,
        "max_text_len": TEXT_LEN,
        "mlm_prob": 0.15,
        "per_gpu_batchsize": BS,
        "num_workers": 0,
        "seed": 0,
        "text_encoder": TEXT_ENC,
        "encoder": TEXT_ENC,
        "model_type": "two-tower",
        "loss_names": {"mlm": 1, "itm": 1},
        "dataset_kwargs": {},
    }
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# Pure dispatch logic
# ---------------------------------------------------------------------------

def test_registry_has_all_documented_names():
    expected = {
        "coco", "coco_karpathy", "f30k", "vg", "gcc", "cc3m", "cc12m",
        "sbu", "vqa", "nlvr2", "snli",
        "refcoco", "refcocoplus", "refcocog", "glue",
    }
    assert expected.issubset(set(_REGISTRY))


def test_nlvr2_has_two_image_keys():
    _, image_keys = _REGISTRY["nlvr2"]
    assert image_keys == ("image_0", "image_1")


def test_glue_has_empty_image_keys():
    _, image_keys = _REGISTRY["glue"]
    assert image_keys == ()


def test_tokenizer_name_picks_text_encoder_for_two_tower():
    cfg = _base_config(model_type="two-tower", text_encoder="bert-base-uncased",
                       encoder="other")
    assert _tokenizer_name(cfg) == "bert-base-uncased"


def test_tokenizer_name_picks_encoder_for_one_tower():
    cfg = _base_config(model_type="one-tower", text_encoder="other",
                       encoder="distilbert-base-uncased")
    assert _tokenizer_name(cfg) == "distilbert-base-uncased"


def test_unknown_dataset_raises():
    with pytest.raises(ValueError, match="Unknown dataset name"):
        build_dataset("bogus", "train", _base_config())


# ---------------------------------------------------------------------------
# Per-dataset kwargs assembly
# ---------------------------------------------------------------------------

def test_kwargs_standard_dataset():
    kw = _kwargs_for("coco", "val", _base_config())
    assert kw == {"seed": 0, "split": "val"}


def test_kwargs_cc3m_includes_streaming():
    kw = _kwargs_for("cc3m", "train", _base_config())
    assert kw["streaming"] is True
    assert kw["split"] == "train"


def test_kwargs_cc3m_streaming_overridable():
    cfg = _base_config(dataset_kwargs={"cc3m": {"streaming": False}})
    kw = _kwargs_for("cc3m", "train", cfg)
    assert kw["streaming"] is False


def test_kwargs_sbu_requires_path():
    with pytest.raises(ValueError, match="data.dataset_kwargs.sbu.path"):
        _kwargs_for("sbu", "train", _base_config())


def test_kwargs_sbu_passes_path():
    cfg = _base_config(dataset_kwargs={"sbu": {"path": "/data/sbu_wds"}})
    kw = _kwargs_for("sbu", "train", cfg)
    assert kw["path"] == "/data/sbu_wds"


def test_kwargs_glue_requires_task():
    with pytest.raises(ValueError, match="data.dataset_kwargs.glue.task"):
        _kwargs_for("glue", "train", _base_config())


def test_kwargs_glue_passes_task_and_split():
    cfg = _base_config(dataset_kwargs={"glue": {"task": "mrpc"}})
    kw = _kwargs_for("glue", "validation", cfg)
    assert kw["task"] == "mrpc"
    assert kw["split"] == "validation"


def test_kwargs_vg_pins_train_split():
    cfg = _base_config(dataset_kwargs={"vg": {"config": "objects_v1.2.0"}})
    kw = _kwargs_for("vg", "anything", cfg)
    assert kw["split"] == "train"
    assert kw["config"] == "objects_v1.2.0"


# ---------------------------------------------------------------------------
# Collator construction from config
# ---------------------------------------------------------------------------

def test_build_collator_respects_loss_names():
    cfg = _base_config(loss_names={"mlm": 0, "itm": 0})
    collator = build_collator(cfg, image_keys=("image",))
    assert collator.do_mlm is False
    assert collator.do_itm is False
    assert collator.image_keys == ("image",)


def test_build_collator_text_only_for_glue():
    cfg = _base_config(loss_names={"mrpc": 1})
    collator = build_collator(cfg, image_keys=())
    assert collator.image_keys == ()


# ---------------------------------------------------------------------------
# End-to-end dataloader path (synthetic data injected by monkeypatching)
# ---------------------------------------------------------------------------

def _synthetic_caption_ds():
    rows = []
    for i in range(BS * 2):
        rows.append({
            "image": {"bytes": _png_bytes((i * 30 % 255, 100, 200)), "path": None},
            "caption": [f"a synthetic caption {i}.{j}" for j in range(5)],
        })
    features = Features({
        "image": DSImage(),
        "caption": Sequence(Value("string")),
    })
    from renaissance.data.transforms import make_vlp_transform
    ds = Dataset.from_list(rows, features=features)
    return ds.with_transform(
        make_vlp_transform(text_column="caption", multi_caption=True, seed=0)
    )


def test_build_dataloader_end_to_end(monkeypatch):
    """Replace `build_dataset` with a synthetic-data stand-in and verify the
    rest of the pipeline (collator + DataLoader) produces a usable batch."""
    ds = _synthetic_caption_ds()

    monkeypatch.setattr(
        "renaissance.data.runner.build_dataset",
        lambda name, split, config: (ds, ("image",)),
    )

    cfg = _base_config(loss_names={"mlm": 0, "itm": 0})
    loader = build_dataloader(cfg, split="train")
    batch = next(iter(loader))

    assert batch["image"][0].shape == (BS, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert batch["text_ids"].shape == (BS, TEXT_LEN)
    assert isinstance(batch["text"], list) and len(batch["text"]) == BS


def test_build_dataloader_rejects_empty_datasets():
    cfg = _base_config(datasets=[])
    with pytest.raises(ValueError, match="at least one dataset"):
        build_dataloader(cfg, split="train")


# ---------------------------------------------------------------------------
# Multi-dataset interleave
# ---------------------------------------------------------------------------

def _synthetic_streaming_wds():
    """Mimic CC3M output (post-loader): IterableDataset of {image: Tensor, text}."""
    features = Features({
        "__key__": Value("string"),
        "jpg": Value("binary"),
        "txt": Value("string"),
    })
    rows = [
        {
            "__key__": f"cc3m/shard0/{i:09d}",
            "jpg": _png_bytes((i * 40 % 255, 50, 150)),
            "txt": f"a cc3m caption {i}",
        }
        for i in range(BS * 2)
    ]
    from renaissance.data.loaders import _WDS_OUT_FEATURES
    from renaissance.data.transforms import make_vlp_transform
    base = Dataset.from_list(rows, features=features)
    ds = base.to_iterable_dataset()
    transform = make_vlp_transform(
        image_columns={"jpg": "image"},
        text_column="txt", multi_caption=False,
    )
    return ds.map(
        transform, batched=True,
        remove_columns=["__key__", "jpg", "txt"],
        features=_WDS_OUT_FEATURES,
    )


def test_interleave_with_two_map_datasets(monkeypatch):
    """Interleave two map-style synthetic datasets (Flickr30k shape).
    Verify normalization yields a uniform `{image, text}` stream that
    flows through the collator."""
    ds_a = _synthetic_caption_ds()
    ds_b = _synthetic_caption_ds()

    def fake_build_dataset(name, split, config):
        return (ds_a if name == "f30k" else ds_b), ("image",)

    monkeypatch.setattr("renaissance.data.runner.build_dataset", fake_build_dataset)

    cfg = _base_config(
        datasets=["f30k", "vg"],  # both use `caption` column
        loss_names={"mlm": 0, "itm": 0},
    )
    loader = build_dataloader(cfg, split="train")
    batch = next(iter(loader))

    assert batch["image"][0].shape == (BS, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert batch["text_ids"].shape == (BS, TEXT_LEN)
    assert isinstance(batch["text"], list) and len(batch["text"]) == BS


def test_interleave_mixes_map_and_streaming(monkeypatch):
    """Map-style (caption) + streaming (CC3M WDS) — interleave should
    produce a unified iterable. This is the realistic pretraining shape
    (e.g. coco + vg + cc3m)."""
    map_ds = _synthetic_caption_ds()
    stream_ds = _synthetic_streaming_wds()

    def fake_build_dataset(name, split, config):
        if name == "f30k":
            return map_ds, ("image",)
        return stream_ds, ("image",)

    monkeypatch.setattr("renaissance.data.runner.build_dataset", fake_build_dataset)

    cfg = _base_config(
        datasets=["f30k", "cc3m"],
        loss_names={"mlm": 0, "itm": 0},
        dataset_probs=[0.5, 0.5],
        stopping_strategy="first_exhausted",
    )
    loader = build_dataloader(cfg, split="train")
    batch = next(iter(loader))

    assert batch["image"][0].shape == (BS, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert batch["text_ids"].shape == (BS, TEXT_LEN)


def test_interleave_strips_task_specific_columns(monkeypatch):
    """After interleave, pass-through columns like img_id should be gone —
    the normalized stream is image+text only."""
    ds = _synthetic_caption_ds()  # has img_id, sentids, filename pass-through

    monkeypatch.setattr(
        "renaissance.data.runner.build_dataset",
        lambda name, split, config: (ds, ("image",)),
    )

    cfg = _base_config(
        datasets=["f30k", "vg"],
        loss_names={"mlm": 0, "itm": 0},
    )
    loader = build_dataloader(cfg, split="train")
    batch = next(iter(loader))

    assert "img_id" not in batch
    assert "sentids" not in batch
    assert "filename" not in batch


def test_interleave_rejects_mismatched_probabilities(monkeypatch):
    monkeypatch.setattr(
        "renaissance.data.runner.build_dataset",
        lambda name, split, config: (_synthetic_caption_ds(), ("image",)),
    )
    cfg = _base_config(
        datasets=["f30k", "vg"],
        dataset_probs=[1.0],  # wrong length
        loss_names={"mlm": 0, "itm": 0},
    )
    with pytest.raises(ValueError, match="dataset_probs has 1 entries"):
        build_dataloader(cfg, split="train")


def test_interleave_rejects_multi_image_dataset(monkeypatch):
    """NLVR2 has image_keys=('image_0','image_1') — can't participate
    in single-image interleave."""
    monkeypatch.setattr(
        "renaissance.data.runner.build_dataset",
        lambda name, split, config: (
            _synthetic_caption_ds(),
            ("image_0", "image_1") if name == "nlvr2" else ("image",),
        ),
    )
    cfg = _base_config(
        datasets=["f30k", "nlvr2"],
        loss_names={"mlm": 0, "itm": 0},
    )
    with pytest.raises(ValueError, match="cannot be interleaved"):
        build_dataloader(cfg, split="train")


def test_interleave_rejects_unsupported_map_dataset(monkeypatch):
    """A map-style dataset without an entry in _INTERLEAVE_TRANSFORM_SPECS
    can't be normalized. Use a fake name to trigger this branch."""
    from renaissance.data.runner import _normalize_for_interleave
    ds = _synthetic_caption_ds()
    with pytest.raises(ValueError, match="no transform spec"):
        _normalize_for_interleave(ds, "fake_name", seed=0)
