"""
End-to-end smoke tests: modern data backend → model forward pass.

These tests build tiny in-memory `datasets.Dataset` objects matching the
schemas the `renaissance.data` loaders emit, run them through the exact
`make_vlp_transform` + `VLPCollator` pipeline `renaissance/data/runner.py`
uses, then feed the collated batch into `RenaissanceModel` and run a
forward pass. The point is to catch wiring breaks between the data layer's
batch schema and what the encoders / objectives expect — the unit tests in
`test_data_modern.py` only exercise the data layer in isolation.

MLM is intentionally not exercised here: `compute_mlm` can produce a NaN
when `DataCollatorForLanguageModeling` happens to mask zero tokens in a
short sequence, which is orthogonal to data-layer wiring. `test_smoke.py`
covers MLM with a controlled batch.

Offline: no Hub fetch, randomly-initialised encoders, CPU-only.
"""

import io
import json

import pytest
import torch
from datasets import Dataset, Features
from datasets import Image as DSImage
from datasets import Sequence, Value
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from renaissance.data import VLPCollator
from renaissance.data.transforms import make_vlp_transform
from renaissance.modeling import RenaissanceModel

from tests.conftest import ALL_LOSS_NAMES, BS, TEXT_LEN


def _png_bytes(color):
    buf = io.BytesIO()
    Image.new("RGB", (48, 64), color=color).save(buf, format="PNG")
    return buf.getvalue()


def _img_field(i):
    return {"bytes": _png_bytes((i * 30 % 255, 100, 200)), "path": None}


def _finite(t):
    return torch.as_tensor(t).isfinite().all().item()


def _tokenizer_for(config):
    name = config["encoder"] if config["model_type"] == "one-tower" else config["text_encoder"]
    return AutoTokenizer.from_pretrained(name)


def _caption_dataset(n):
    rows = [
        {"image": _img_field(i), "caption": [f"a caption {i}.{j}" for j in range(5)]}
        for i in range(n)
    ]
    features = Features({"image": DSImage(), "caption": Sequence(Value("string"))})
    ds = Dataset.from_list(rows, features=features)
    return ds.with_transform(
        make_vlp_transform(text_column="caption", multi_caption=True, seed=0)
    )


def _collate(ds, config, *, do_itm, image_keys=("image",)):
    """Build a non-MLM collator the way runner.build_collator would and
    return one batch. do_mlm is always False — see module docstring."""
    tok = _tokenizer_for(config)
    collator = VLPCollator(
        tok,
        max_text_len=TEXT_LEN,
        do_mlm=False,
        do_itm=do_itm,
        image_keys=image_keys,
        image_size=config["image_size"] if image_keys else None,
    )
    loader = DataLoader(ds, batch_size=BS, collate_fn=collator)
    return next(iter(loader))


# ---------------------------------------------------------------------------
# Encoder consumes the collated batch
# ---------------------------------------------------------------------------

def test_two_tower_infer_from_collated_batch(two_tower_pretrain_config):
    batch = _collate(_caption_dataset(BS), two_tower_pretrain_config, do_itm=False)
    model = RenaissanceModel(two_tower_pretrain_config)
    model.eval()
    with torch.no_grad():
        out = model.infer(batch)
    assert out["cls_feats"].shape[0] == BS
    assert _finite(out["cls_feats"])


def test_one_tower_infer_from_collated_batch(one_tower_pretrain_config):
    batch = _collate(_caption_dataset(BS), one_tower_pretrain_config, do_itm=False)
    model = RenaissanceModel(one_tower_pretrain_config)
    model.eval()
    with torch.no_grad():
        out = model.infer(batch)
    assert out["cls_feats"].shape[0] == BS
    assert _finite(out["cls_feats"])


# ---------------------------------------------------------------------------
# ITM: collator's in-batch negative (`false_image_0`) feeds compute_itm
# ---------------------------------------------------------------------------

@pytest.fixture
def itm_config(two_tower_pretrain_config):
    cfg = dict(two_tower_pretrain_config)
    cfg["loss_names"] = {**ALL_LOSS_NAMES, "itm": 1}
    return cfg


def test_two_tower_itm_e2e(itm_config):
    batch = _collate(_caption_dataset(BS), itm_config, do_itm=True)
    assert "false_image_0" in batch
    model = RenaissanceModel(itm_config)
    model.train()
    model.current_tasks = ["itm"]
    ret = model(batch)
    assert _finite(ret["itm_loss"])


# ---------------------------------------------------------------------------
# VQA: pass-through label columns feed compute_vqa
# ---------------------------------------------------------------------------

@pytest.fixture
def vqa_config(two_tower_pretrain_config):
    cfg = dict(two_tower_pretrain_config)
    cfg["loss_names"] = {**ALL_LOSS_NAMES, "vqa": 1}
    cfg["vqav2_label_size"] = 4
    return cfg


def test_two_tower_vqa_e2e(vqa_config, tmp_path):
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(json.dumps(["red", "blue", "green", "yellow"]))

    from renaissance.data.vqa import answers_to_labels_scores, load_vqa_answer_vocab
    _, ans2label = load_vqa_answer_vocab(str(vocab_path))

    colors = ["red", "blue", "green", "yellow"]
    rows = [
        {
            "image": _img_field(i),
            "question": f"what color is object {i}?",
            "answers": [{"answer": colors[i]}] * 10,
        }
        for i in range(BS)
    ]
    features = Features({
        "image": DSImage(),
        "question": Value("string"),
        "answers": [{"answer": Value("string")}],
    })
    ds = Dataset.from_list(rows, features=features)
    base_tf = make_vlp_transform(text_column="question", multi_caption=False)

    def transform(batch):
        out = base_tf(batch)
        labels, scores = [], []
        for ann in batch["answers"]:
            ls, ss = answers_to_labels_scores(ann, ans2label)
            labels.append(ls)
            scores.append(ss)
        out["vqa_labels"] = labels
        out["vqa_scores"] = scores
        return out

    ds = ds.with_transform(transform)
    batch = _collate(ds, vqa_config, do_itm=False)
    assert "vqa_labels" in batch and "vqa_scores" in batch

    model = RenaissanceModel(vqa_config)
    model.train()
    model.current_tasks = ["vqa"]
    ret = model(batch)
    assert _finite(ret["vqa_loss"])
    assert ret["vqa_logits"].shape == (BS, 4)
