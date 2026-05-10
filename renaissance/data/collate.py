"""
Batch-level collator for the modern data layer.

Takes a list of per-item dicts (as produced by `Dataset.__getitem__` after
`with_transform`) and produces the batch dict the existing model encoders
expect:

- `image`: list[Tensor[B, 3, H, W]]  (single-view list for legacy compat)
- `text`: list[str] of length B
- `text_ids`, `text_masks`, `text_labels`: Tensor[B, L]
- `text_ids_mlm`, `text_labels_mlm`: Tensor[B, L]  (when `do_mlm=True`)
- `false_image_0`: list[Tensor[B, 3, H, W]]       (when `do_itm=True`)

ITM negatives are produced via an in-batch image permutation rather than per-
item random sampling. MLM masking uses `DataCollatorForLanguageModeling`.
"""

import torch


class VLPCollator:
    def __init__(
        self,
        tokenizer,
        max_text_len: int = 40,
        mlm_prob: float = 0.15,
        do_mlm: bool = True,
        do_itm: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.do_mlm = do_mlm
        self.do_itm = do_itm
        if do_mlm:
            # Imported lazily: `transformers.data.data_collator` pulls in a TF
            # module that breaks in environments where TF + numpy are
            # mismatched. Non-MLM users (classifier fine-tuners) shouldn't pay
            # for that import.
            from transformers import DataCollatorForLanguageModeling

            self.mlm_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=True,
                mlm_probability=mlm_prob,
                return_tensors="pt",
            )
        else:
            self.mlm_collator = None

    def __call__(self, examples):
        images = torch.stack([ex["image"] for ex in examples])
        texts = [ex["text"] for ex in examples]

        enc = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )

        batch = {
            "image": [images],
            "text": texts,
            "text_ids": enc["input_ids"],
            "text_masks": enc["attention_mask"],
            "text_labels": torch.full_like(enc["input_ids"], -100),
        }

        if self.do_mlm:
            mlm_inputs = [
                {
                    "input_ids": enc["input_ids"][i].tolist(),
                    "attention_mask": enc["attention_mask"][i].tolist(),
                    "special_tokens_mask": enc["special_tokens_mask"][i].tolist(),
                }
                for i in range(len(texts))
            ]
            mlm_out = self.mlm_collator(mlm_inputs)
            batch["text_ids_mlm"] = mlm_out["input_ids"]
            batch["text_labels_mlm"] = mlm_out["labels"]

        if self.do_itm:
            perm = torch.randperm(images.size(0))
            batch["false_image_0"] = [images[perm].clone()]

        # Pass-through for any task-specific columns (qid, labels, bbox, ...)
        for ex in examples:
            for k, v in ex.items():
                if k in ("image", "text"):
                    continue
                batch.setdefault(k, []).append(v)

        return batch
