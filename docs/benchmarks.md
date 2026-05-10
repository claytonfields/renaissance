# Benchmarks

Expected metric values on standard validation splits.  These numbers serve as
regression targets — if a newly trained checkpoint scores below them, it is a
signal to investigate training configuration, data preprocessing, or bugs.

Numbers were produced with:
```bash
python -m renaissance.eval \
    --checkpoint <checkpoint_path> \
    --task <task> \
    --split val \
    --data_root data/arrow/
```

---

## Image–Text Pretraining (Two-Tower)

| Metric | Val |
|--------|-----|
| ITM accuracy | — |
| MLM accuracy | — |

---

## Visual Question Answering (VQAv2)

Model: two-tower pretrained on COCO+VG, fine-tuned on VQAv2.

| Split | VQA Score |
|-------|-----------|
| val   | — |
| test-dev | — |

Evaluation note: VQA score is the soft accuracy that accounts for human
annotator disagreement (score = min(count / 3, 1.0) per answer).

---

## NLVR2

| Split | Accuracy |
|-------|----------|
| dev   | — |
| test-P | — |

---

## SNLI-VE

| Split | Accuracy |
|-------|----------|
| dev   | — |
| test  | — |

---

## Image–Text Retrieval (IRTR)

IRTR recall metrics require the two-pass `compute_irtr_recall` function with
dedicated text and image dataloaders (not the standard batch-level evaluator).

| Direction | R@1 | R@5 | R@10 |
|-----------|-----|-----|------|
| Image→Text | — | — | — |
| Text→Image | — | — | — |

---

## RefCOCO (Reference Resolution)

Accuracy@0.5 IoU: predicted bounding box IoU with ground truth ≥ 0.5.

| Split | Accuracy |
|-------|----------|
| val   | — |
| testA | — |
| testB | — |

---

## GLUE (text-only tasks, one-tower)

| Task | Metric | Score |
|------|--------|-------|
| MRPC | F1 / Accuracy | — / — |

---

*Numbers marked — will be filled in once pretrained checkpoints are released.*
