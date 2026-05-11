"""
VQAv2 answer-vocabulary utilities for the modern data layer.

VQAv2 is trained as multi-label classification over a fixed answer vocab
(the 3129 answers that appear >= 9 times as the `multiple_choice_answer`
across the train+val annotations, the standard METER/ViLT/BLIP recipe).
The training target for a question is a soft multi-hot vector: each label
the 10 annotators gave gets a score on the 0/0.3/0.6/0.9/1.0 scale based
on how many annotators said it.

The 3129-answer vocab is bundled as `vqa_answer_vocab.json` next to this
file. Pass a custom vocab path to `load_vqav2` / `load_vqa_answer_vocab`
if you need one matching a pretrained checkpoint's classifier head.
"""

import json
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

VQA_DEFAULT_VOCAB_PATH = os.path.join(os.path.dirname(__file__), "vqa_answer_vocab.json")


def get_vqa_score(occurrences: int) -> float:
    """VQAv2 soft accuracy for an answer the annotators gave `occurrences`
    times: 0→0.0, 1→0.3, 2→0.6, 3→0.9, 4+→1.0. (Explicit table to match the
    legacy build exactly and avoid float drift from `occurrences * 0.3`.)"""
    return {0: 0.0, 1: 0.3, 2: 0.6, 3: 0.9}.get(occurrences, 1.0)


def load_vqa_answer_vocab(
    path: Optional[str] = None,
) -> Tuple[List[str], Dict[str, int]]:
    """Load the answer vocab. Returns ``(label2ans, ans2label)``.

    Default: the 3129-answer vocab bundled with the package. ``path`` may
    point at a JSON list of answer strings (label index = position).
    """
    path = path or VQA_DEFAULT_VOCAB_PATH
    with open(path, "r", encoding="utf-8") as f:
        label2ans = json.load(f)
    ans2label = {a: i for i, a in enumerate(label2ans)}
    return label2ans, ans2label


def build_vqa_answer_vocab(major_answers, min_occurrences: int = 9) -> List[str]:
    """Build the answer vocab from an iterable of `multiple_choice_answer`
    strings (one per question): normalize, count, keep answers seen at least
    ``min_occurrences`` times. Returns the ordered ``label2ans`` list.

    Use this when bringing your own VQAv2 annotations (e.g. because the
    lmms-lab Hub mirror doesn't host the train split).
    """
    from renaissance.utils.glossary import normalize_word

    counter = Counter(normalize_word(a) for a in major_answers)
    return [a for a, c in counter.items() if c >= min_occurrences]


def _annotator_answer_strings(answers) -> List[str]:
    """Coerce the various HF representations of a question's 10 annotator
    answers into a flat list of answer strings.

    Handles:
    - list of strings: ``["red", "red", ...]``
    - list of dicts (raw VQAv2 format): ``[{"answer": "red", ...}, ...]``
    - struct-of-arrays (HF ``Sequence(struct)``): ``{"answer": ["red", ...], ...}``
    """
    if isinstance(answers, dict):
        if "answer" in answers:
            return list(answers["answer"])
        raise ValueError(f"Unrecognized VQA answers dict keys: {sorted(answers)}")
    return [a["answer"] if isinstance(a, dict) else a for a in answers]


def answers_to_labels_scores(
    answers, ans2label: Dict[str, int],
) -> Tuple[List[int], List[float]]:
    """Map a question's 10 annotator answers to ``(labels, scores)`` —
    parallel lists usable by ``objectives.compute_vqa`` to build the soft
    target. Answers not in the vocab are dropped.
    """
    from renaissance.utils.glossary import normalize_word

    counts = Counter(normalize_word(a) for a in _annotator_answer_strings(answers))
    labels: List[int] = []
    scores: List[float] = []
    for ans, cnt in counts.items():
        idx = ans2label.get(ans)
        if idx is not None:
            labels.append(idx)
            scores.append(get_vqa_score(cnt))
    return labels, scores
