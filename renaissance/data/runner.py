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

from datasets import IterableDataset
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
    """Assemble the kwargs to pass to a loader. CC3M/CC12M and SBU take
    positional-style image_size + special args; everything else is uniform."""
    base = {"image_size": config["image_size"], "seed": config.get("seed", 0)}
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
        base.pop("image_size", None)  # GLUE is text-only
        base["task"] = extra["task"]
        base["split"] = split
        return base

    if name == "vg":
        if "config" in extra:
            base["config"] = extra["config"]
        base["split"] = "train"  # VG has only train
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
    - max_text_len, mlm_prob
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
    )


def build_dataloader(config: Dict[str, Any], split: str) -> DataLoader:
    """Top-level builder used by run.py. Single-dataset only — multi-dataset
    interleave (replacement for the legacy ConcatDataset path) is a follow-up.
    """
    datasets = config.get("datasets", [])
    if len(datasets) != 1:
        raise NotImplementedError(
            f"modern backend currently supports a single dataset; got {datasets!r}. "
            "Multi-dataset interleave via datasets.interleave_datasets is a TODO."
        )
    name = datasets[0]
    ds, image_keys = build_dataset(name, split, config)
    collator = build_collator(config, image_keys)
    return _wrap_dataloader(ds, collator, config, split)


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
