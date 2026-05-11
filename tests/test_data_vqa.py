"""
Tests for the modern VQA answer-vocab utilities (`renaissance.data.vqa`)
and the `load_vqav2` answers → labels/scores mapping.

Offline — uses the bundled vocab and small synthetic datasets.
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

from renaissance.data import VLPCollator
from renaissance.data.vqa import (
    answers_to_labels_scores,
    build_vqa_answer_vocab,
    get_vqa_score,
    load_vqa_answer_vocab,
)
from renaissance.data.transforms import make_vlp_transform

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


# ---------------------------------------------------------------------------
# Score table
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("occ,expected", [(0, 0.0), (1, 0.3), (2, 0.6), (3, 0.9), (4, 1.0), (10, 1.0)])
def test_get_vqa_score(occ, expected):
    assert get_vqa_score(occ) == expected


# ---------------------------------------------------------------------------
# Bundled vocab
# ---------------------------------------------------------------------------

def test_bundled_vocab_has_3129_entries():
    label2ans, ans2label = load_vqa_answer_vocab()
    assert len(label2ans) == 3129
    assert len(ans2label) == 3129
    # ans2label is the inverse of label2ans
    assert all(ans2label[label2ans[i]] == i for i in range(0, 3129, 137))
    # canonical sanity checks
    assert "yes" in ans2label and "no" in ans2label


def test_load_vocab_from_custom_path(tmp_path):
    import json
    p = tmp_path / "vocab.json"
    p.write_text(json.dumps(["yes", "no", "maybe"]))
    label2ans, ans2label = load_vqa_answer_vocab(str(p))
    assert label2ans == ["yes", "no", "maybe"]
    assert ans2label == {"yes": 0, "no": 1, "maybe": 2}


# ---------------------------------------------------------------------------
# Vocab building from major answers
# ---------------------------------------------------------------------------

def test_build_vocab_threshold_and_normalization():
    # "yes" appears 4×, "no" 3×, "maybe" 1×; threshold 3 → keep yes, no.
    # "Yes!" normalizes to "yes" so it counts toward the "yes" tally.
    major = ["yes", "Yes!", "yes", "yes"] + ["no", "no", "no"] + ["maybe"]
    vocab = build_vqa_answer_vocab(major, min_occurrences=3)
    assert set(vocab) == {"yes", "no"}


# ---------------------------------------------------------------------------
# answers → (labels, scores)
# ---------------------------------------------------------------------------

def test_answers_to_labels_scores_list_of_strings():
    ans2label = {"red": 0, "blue": 1, "green": 2}
    # 6× red, 3× blue, 1× green → labels [0,1,2] with scores [1.0, 0.9, 0.3]
    answers = ["red"] * 6 + ["blue"] * 3 + ["green"]
    labels, scores = answers_to_labels_scores(answers, ans2label)
    by_label = dict(zip(labels, scores))
    assert by_label == {0: 1.0, 1: 0.9, 2: 0.3}


def test_answers_to_labels_scores_drops_unknown():
    ans2label = {"red": 0}
    labels, scores = answers_to_labels_scores(["red"] * 5 + ["purple"] * 5, ans2label)
    assert labels == [0] and scores == [1.0]


def test_answers_to_labels_scores_list_of_dicts():
    """Raw VQAv2 annotation format: list of {"answer": ..., ...} dicts."""
    ans2label = {"red": 0, "blue": 1}
    answers = [{"answer": "red", "answer_confidence": "yes"}] * 7 + [{"answer": "blue"}] * 3
    labels, scores = answers_to_labels_scores(answers, ans2label)
    assert dict(zip(labels, scores)) == {0: 1.0, 1: 0.9}


def test_answers_to_labels_scores_struct_of_arrays():
    """HF Sequence(struct) representation: {"answer": [...], ...}."""
    ans2label = {"red": 0, "blue": 1}
    answers = {"answer": ["red"] * 8 + ["blue"] * 2, "answer_confidence": ["yes"] * 10}
    labels, scores = answers_to_labels_scores(answers, ans2label)
    assert dict(zip(labels, scores)) == {0: 1.0, 1: 0.6}


# ---------------------------------------------------------------------------
# End-to-end: load_vqav2 transform produces vqa_labels / vqa_scores
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_vqav2_with_answers(tmp_path):
    """VQAv2-shaped Dataset with a small custom vocab path so the test
    doesn't depend on the bundled 3129 vocab."""
    import json
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(json.dumps(["red", "blue", "green", "yellow"]))

    colors = ["red", "blue", "green", "yellow"]
    rows = []
    for i in range(BS):
        # i+1 annotators say colors[i], the rest say "unknownish"
        n = i + 1
        answers = [{"answer": colors[i]}] * n + [{"answer": "unknownish"}] * (10 - n)
        rows.append({
            "image": {"bytes": _png_bytes((i * 30 % 255, 100, 200)), "path": None},
            "question": f"what color is object {i}?",
            "question_id": 1000 + i,
            "multiple_choice_answer": colors[i],
            "answers": answers,
            "image_id": i,
        })
    features = Features({
        "image": DSImage(),
        "question": Value("string"),
        "question_id": Value("int64"),
        "multiple_choice_answer": Value("string"),
        "answers": [{"answer": Value("string")}],
        "image_id": Value("int64"),
    })
    ds = Dataset.from_list(rows, features=features)

    # Replicate what load_vqav2 does internally (can't call it directly —
    # it would hit the Hub).
    from renaissance.data.vqa import answers_to_labels_scores, load_vqa_answer_vocab
    _, ans2label = load_vqa_answer_vocab(str(vocab_path))
    base_tf = make_vlp_transform(text_column="question", multi_caption=False)

    def transform(batch):
        out = base_tf(batch)
        out["qid"] = list(out["question_id"])
        labels, scores = [], []
        for ann in batch["answers"]:
            ls, ss = answers_to_labels_scores(ann, ans2label)
            labels.append(ls)
            scores.append(ss)
        out["vqa_labels"] = labels
        out["vqa_scores"] = scores
        return out

    return ds.with_transform(transform)


def test_load_vqav2_transform_produces_labels_and_scores(synthetic_vqav2_with_answers, tokenizer):
    collator = VLPCollator(
        tokenizer, max_text_len=TEXT_LEN, do_mlm=False, do_itm=False, image_size=IMAGE_SIZE,
    )
    loader = DataLoader(synthetic_vqav2_with_answers, batch_size=BS, collate_fn=collator)
    batch = next(iter(loader))

    assert batch["image"][0].shape == (BS, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert batch["text_ids"].shape == (BS, TEXT_LEN)
    assert batch["qid"] == [1000, 1001, 1002, 1003]

    # Row i: colors[i] said (i+1) times → label i with score get_vqa_score(i+1).
    # "unknownish" isn't in the vocab so it's dropped.
    assert len(batch["vqa_labels"]) == BS
    for i in range(BS):
        assert batch["vqa_labels"][i] == [i]
        assert batch["vqa_scores"][i] == [get_vqa_score(i + 1)]

    # Sanity: these are exactly what compute_vqa needs to build the soft target.
    label_size = 4
    target = torch.zeros(BS, label_size)
    for i, (labs, scs) in enumerate(zip(batch["vqa_labels"], batch["vqa_scores"])):
        for l, s in zip(labs, scs):
            target[i, l] = s
    assert torch.allclose(target.diagonal(), torch.tensor([get_vqa_score(i + 1) for i in range(BS)]))
