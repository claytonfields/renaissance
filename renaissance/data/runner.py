"""
Entry point that translates a flat Renaissance config dict into a
PyTorch DataLoader backed by the modern `renaissance.data` layer.

Used by `run.py` when `data.backend == "modern"`.

Dispatch uses the same short dataset names as the legacy `MTDataModule`
(``coco``, ``f30k``, ``vg``, ``vqa``, ``nlvr2``, ``snli``, ``gcc``,
``sbu``, ``glue``, ``refcoco``) plus the new modern-only names
(``coco_karpathy``, ``cc3m``, ``cc12m``, ``refcocoplus``, ``refcocog``).
"""

from typing import Any, Dict, Tuple

from datasets import Features, IterableDataset, interleave_datasets
from datasets import Image as DSImage
from datasets import Value as DSValue
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .collate import VLPCollator
from .loaders import (
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
)
from .transforms import make_vlp_transform

# Maps a dataset short-name to (loader_fn, image_keys_tuple).
# Loaders are called with kwargs assembled from the flat config dict.
_REGISTRY: Dict[str, Tuple[Any, Tuple[str, ...]]] = {
    "coco": (load_coco_karpathy, ("image",)),
    "coco_karpathy": (load_coco_karpathy, ("image",)),
    "f30k": (load_flickr30k, ("image",)),
    "vg": (load_visual_genome, ("image",)),
    "gcc": (load_cc3m, ("image",)),
    "cc3m": (load_cc3m, ("image",)),
    "cc12m": (load_cc12m, ("image",)),
    "sbu": (load_sbu, ("image",)),
    "vqa": (load_vqav2, ("image",)),
    "nlvr2": (load_nlvr2, ("image_0", "image_1")),
    "snli": (load_snli_ve, ("image",)),
    "refcoco": (load_refcoco, ("image",)),
    "refcocoplus": (load_refcocoplus, ("image",)),
    "refcocog": (load_refcocog, ("image",)),
    "glue": (load_glue, ()),
}


def _tokenizer_name(config: Dict[str, Any]) -> str:
    """Pick the right HF model id for tokenization based on model_type."""
    if config.get("model_type") == "one-tower":
        return config["encoder"]
    return config.get("text_encoder", config.get("encoder"))


def _kwargs_for(name: str, split: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble the kwargs to pass to a loader. Image tensorization now
    happens in the collator, so loaders no longer take `image_size`."""
    base = {"seed": config.get("seed", 0)}
    extra = config.get("dataset_kwargs", {}).get(name, {})

    if name in ("cc3m", "cc12m", "gcc"):
        # WDS loaders take split + streaming
        base["split"] = "train" if name in ("cc12m",) else split
        base["streaming"] = extra.get("streaming", True)
        if "hub_id" in extra:
            base["hub_id"] = extra["hub_id"]
        return base

    if name == "sbu":
        if "path" not in extra:
            raise ValueError(
                "load_sbu requires data.dataset_kwargs.sbu.path "
                "(local img2dataset WebDataset directory)."
            )
        base["path"] = extra["path"]
        return base

    if name == "glue":
        if "task" not in extra:
            raise ValueError(
                "load_glue requires data.dataset_kwargs.glue.task "
                "(one of mrpc, rte, mnli, sst2, cola, qnli, qqp, wnli, stsb)."
            )
        base["task"] = extra["task"]
        base["split"] = split
        return base

    if name == "vg":
        if "config" in extra:
            base["config"] = extra["config"]
        base["split"] = "train"  # VG has only train
        return base

    if name == "vqa":
        if "answer_vocab" in extra:
            base["answer_vocab"] = extra["answer_vocab"]
        base["split"] = split
        return base

    # Standard image-text datasets take a `split`.
    base["split"] = split
    return base


def build_dataset(name: str, split: str, config: Dict[str, Any]):
    """Resolve and call the right loader. Returns (ds, image_keys)."""
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown dataset name {name!r}. Known: {sorted(_REGISTRY)}"
        )
    loader_fn, image_keys = _REGISTRY[name]
    kwargs = _kwargs_for(name, split, config)
    ds = loader_fn(**kwargs)
    return ds, image_keys


def build_collator(config: Dict[str, Any], image_keys: Tuple[str, ...]) -> VLPCollator:
    """Construct a VLPCollator from a flat config dict.

    Reads:
    - text_encoder / encoder (whichever matches model_type) for tokenizer
    - max_text_len, mlm_prob, image_size
    - loss_names.mlm and loss_names.itm to gate MLM masking and ITM negatives
    """
    tokenizer = AutoTokenizer.from_pretrained(_tokenizer_name(config))
    loss_names = config.get("loss_names", {})
    return VLPCollator(
        tokenizer=tokenizer,
        max_text_len=config.get("max_text_len", 40),
        mlm_prob=config.get("mlm_prob", 0.15),
        do_mlm=bool(loss_names.get("mlm", 0)),
        do_itm=bool(loss_names.get("itm", 0)),
        image_keys=image_keys,
        image_size=config["image_size"] if image_keys else None,
    )


def build_dataloader(config: Dict[str, Any], split: str) -> DataLoader:
    """Top-level builder used by run.py.

    - 1 dataset: pass through, all task-specific columns preserved.
    - >1 datasets: use `datasets.interleave_datasets` after normalizing each
      to a {image, text}-only IterableDataset. Multi-image (NLVR2) and
      text-only (GLUE) datasets cannot participate.
    """
    datasets_arg = config.get("datasets", [])
    if not datasets_arg:
        raise ValueError("config.datasets must contain at least one dataset")

    if len(datasets_arg) == 1:
        name = datasets_arg[0]
        ds, image_keys = build_dataset(name, split, config)
        collator = build_collator(config, image_keys)
        return _wrap_dataloader(ds, collator, config, split)

    return build_interleaved_dataloader(config, split)


# ---------------------------------------------------------------------------
# Multi-dataset interleave
# ---------------------------------------------------------------------------

# Per-dataset transform configuration when participating in interleave. Mirrors
# the args that each loader passes to `make_vlp_transform`. Datasets not listed
# here are unsupported for interleave (NLVR2: multi-image; GLUE: text-only;
# streaming WDS loaders: already apply their transform via .map).
_INTERLEAVE_TRANSFORM_SPECS: Dict[str, Dict[str, Any]] = {
    "coco":           dict(text_column="captions", multi_caption=True),
    "coco_karpathy":  dict(text_column="captions", multi_caption=True),
    "f30k":           dict(text_column="caption",  multi_caption=True),
    "vg":             dict(text_column="caption",  multi_caption=True),
    "vqa":            dict(text_column="question", multi_caption=False),
    "snli":           dict(text_column="hypothesis", multi_caption=False),
    "refcoco":        dict(text_column="question", multi_caption=False),
    "refcocoplus":    dict(text_column="question", multi_caption=False),
    "refcocog":       dict(text_column="question", multi_caption=False),
}


def _normalize_for_interleave(ds, name: str, seed: int) -> IterableDataset:
    """Convert one dataset to an IterableDataset whose iteration yields
    `{image, text}` only — image is PIL, text is str (no tensors so the
    schema stays Arrow-compatible for interleave_datasets feature inference).

    Map-style loaders attach the transform via `with_transform`, which doesn't
    survive `.to_iterable_dataset()`. We clear it with `with_format(None)`,
    convert to iterable, and re-apply the transform via `.map` with
    `pass_through=False` so the output schema is uniform across all
    participants.

    Streaming loaders (CC3M, CC12M, SBU) already produce `{image, text}` via
    `.map`; we just trim any stray columns.
    """
    if isinstance(ds, IterableDataset):
        cols = ds.column_names or []
        if cols and any(c not in ("image", "text") for c in cols):
            ds = ds.select_columns(["image", "text"])
        return ds

    if name not in _INTERLEAVE_TRANSFORM_SPECS:
        raise ValueError(
            f"Dataset {name!r} cannot be interleaved — no transform spec. "
            "Only single-image caption-style datasets are interleavable; "
            "NLVR2 and GLUE have incompatible schemas."
        )
    transform = make_vlp_transform(
        seed=seed,
        pass_through=False,
        **_INTERLEAVE_TRANSFORM_SPECS[name],
    )

    ds = ds.with_format(None).to_iterable_dataset()
    # Drop source columns that the transform doesn't re-emit; `select_columns`
    # afterwards guarantees the iterated rows are exactly {image, text}
    # regardless of `datasets`-version quirks.
    drop = [c for c in ds.column_names if c not in ("image", "text")]
    ds = ds.map(
        transform,
        batched=True,
        remove_columns=drop or None,
        features=Features({"image": DSImage(), "text": DSValue("string")}),
    )
    return ds.select_columns(["image", "text"])


def build_interleaved_dataloader(
    config: Dict[str, Any], split: str,
) -> DataLoader:
    names = config["datasets"]
    probs = config.get("dataset_probs")
    stopping = config.get("stopping_strategy", "first_exhausted")
    seed = config.get("seed", 0)

    if probs is not None and len(probs) != len(names):
        raise ValueError(
            f"dataset_probs has {len(probs)} entries but datasets has "
            f"{len(names)} — must match."
        )

    iterables = []
    for name in names:
        ds, image_keys = build_dataset(name, split, config)
        if image_keys != ("image",):
            raise ValueError(
                f"Dataset {name!r} with image_keys={image_keys} cannot be "
                "interleaved (multi-image or text-only datasets aren't compatible)."
            )
        iterables.append(_normalize_for_interleave(ds, name, seed))

    interleaved = interleave_datasets(
        iterables,
        probabilities=probs,
        stopping_strategy=stopping,
        seed=seed,
    )
    collator = build_collator(config, image_keys=("image",))
    return _wrap_dataloader(interleaved, collator, config, split)


def _wrap_dataloader(ds, collator, config: Dict[str, Any], split: str) -> DataLoader:
    """DataLoader wrapping with shuffle/workers/pin_memory wired from config.
    IterableDataset (streaming) sidesteps shuffle since it has no random
    access."""
    is_streaming = isinstance(ds, IterableDataset)
    return DataLoader(
        ds,
        batch_size=config.get("per_gpu_batchsize", 32),
        shuffle=(split == "train" and not is_streaming),
        num_workers=config.get("num_workers", 0),
        collate_fn=collator,
        pin_memory=True,
    )
