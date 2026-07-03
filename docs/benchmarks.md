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

Two-tower, image_size=384, deit-tiny + electra-small, `cross_layer_hidden_size=256`, `num_cross_layers=6`. Pretrained MLM+ITM then fine-tuned on SNLI-VE.

| Split | Accuracy (Lightning, 2024-09) | Accuracy (rewrite, `scripts/regress_snli_ckpt.py`) |
|-------|-------------------------------|-----------------------------------------------------|
| dev   | 0.7405 | 0.7283 |
| test  | 0.7455 | 0.7279 |

The "rewrite" column is a regression run of the same weights (2024‑09 fine-tune ckpt) through the current `RenaissanceModel` code path; `strict=True` load matches all 666 parameters (see `scripts/regress_snli_ckpt.py` for the two-rule rename). Residual ~1.5 pp gap is attributed to environment drift (transformers/torch/torchvision updates since 2024‑09).

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
