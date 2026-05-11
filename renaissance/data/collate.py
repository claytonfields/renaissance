"""
Batch-level collator for the modern data layer.

Takes a list of per-item dicts (from `Dataset.__getitem__` after
`with_transform`) and produces the batch dict the existing encoders expect.

Standard single-image schema (`image_keys=("image",)`):
- `image`: list[Tensor[B, 3, H, W]]  (single-view list, legacy compat)
- `text`: list[str] of length B
- `text_ids`, `text_masks`, `text_labels`: Tensor[B, L]
- `text_ids_mlm`, `text_labels_mlm`: Tensor[B, L]  (when `do_mlm=True`)
- `false_image_0`: list[Tensor[B, 3, H, W]]       (when `do_itm=True`)

NLVR2-style two-image schema (`image_keys=("image_0", "image_1")`):
- `image_0`, `image_1`: each a list[Tensor[B, 3, H, W]]
- text fields as above

Text-only schema for GLUE (`image_keys=()`):
- No image fields produced; ITM silently no-ops.
- If examples carry a `text_pair` field, the tokenizer encodes
  `(text, text_pair)` as a pair (sentence A / sentence B).

ITM negatives only fire when `do_itm=True` AND `image_keys` is non-empty;
they use the first key in `image_keys` as the source.
"""

from typing import Optional, Tuple

import torch

from .transforms import _to_pil, make_image_transform


class VLPCollator:
    def __init__(
        self,
        tokenizer,
        max_text_len: int = 40,
        mlm_prob: float = 0.15,
        do_mlm: bool = True,
        do_itm: bool = True,
        image_keys: Tuple[str, ...] = ("image",),
        image_size: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.do_mlm = do_mlm
        self.do_itm = do_itm
        self.image_keys = tuple(image_keys)
        self.image_size = image_size
        # Final image transform applied at batch time. None is allowed only
        # for fully text-only collators (image_keys=()).
        if image_keys and image_size is None:
            raise ValueError(
                "image_size must be set when image_keys is non-empty. "
                "Pass image_size=N to VLPCollator."
            )
        self.image_transform = make_image_transform(image_size) if image_size else None
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
        batch = {}

        for k in self.image_keys:
            tensors = []
            for ex in examples:
                img = ex[k]
                if isinstance(img, torch.Tensor):
                    # Already tensorized upstream (legacy code path).
                    tensors.append(img)
                else:
                    tensors.append(self.image_transform(_to_pil(img)))
            batch[k] = [torch.stack(tensors)]

        texts = [ex["text"] for ex in examples]
        text_pairs = (
            [ex["text_pair"] for ex in examples]
            if examples and "text_pair" in examples[0]
            else None
        )

        if text_pairs is not None:
            enc = self.tokenizer(
                texts,
                text_pair=text_pairs,
                padding="max_length",
                truncation=True,
                max_length=self.max_text_len,
                return_special_tokens_mask=True,
                return_tensors="pt",
            )
        else:
            enc = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_text_len,
                return_special_tokens_mask=True,
                return_tensors="pt",
            )

        batch["text"] = texts
        if text_pairs is not None:
            batch["text_pair"] = text_pairs
        batch["text_ids"] = enc["input_ids"]
        batch["text_masks"] = enc["attention_mask"]
        batch["text_labels"] = torch.full_like(enc["input_ids"], -100)

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

        if self.do_itm and self.image_keys:
            primary = self.image_keys[0]
            primary_tensor = batch[primary][0]
            perm = torch.randperm(primary_tensor.size(0))
            batch[f"false_{primary}_0"] = [primary_tensor[perm].clone()]

        skip = set(self.image_keys) | {"text", "text_pair"}
        for ex in examples:
            for k, v in ex.items():
                if k in skip:
                    continue
                batch.setdefault(k, []).append(v)

        return batch
