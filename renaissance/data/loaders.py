"""
Per-dataset loaders. Each function returns a `datasets.Dataset` (or
`IterableDataset` for streaming sources) with `with_transform` applied so
each row yields `{image: Tensor[3, H, W], text: str, ...}` (or
`{image_0, image_1, text, ...}` for two-image datasets).

Pair these with `VLPCollator` and a standard `torch.utils.data.DataLoader`.

Dataset-specific label processing (VQA answer-vocab mapping, RefCOCO bbox
normalization, etc.) is deferred to the evaluator / objective so loaders
stay thin.
"""

from typing import Optional

from datasets import Features
from datasets import Image as DSImage
from datasets import Value as DSValue
from datasets import load_dataset

# Features declaration for streaming WDS .map output. Explicit so HF doesn't
# fall back to Arrow type inference on PIL.Image (which fails).
_WDS_OUT_FEATURES = Features({"image": DSImage(), "text": DSValue("string")})

from .transforms import make_vlp_transform


def load_flickr30k(
    split: str,
    hub_id: str = "nlphuji/flickr30k",
    seed: Optional[int] = None,
):
    """Karpathy-split Flickr30k from `nlphuji/flickr30k`.

    All 31k images live in a single "test" split with a `split` column
    tagging Karpathy assignment. Each row has a `caption` column with 5
    strings.
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be one of train/val/test, got {split!r}")

    ds = load_dataset(hub_id, split="test")
    ds = ds.filter(lambda x: x["split"] == split)
    ds = ds.cast_column("image", DSImage(decode=True))
    ds = ds.with_transform(
        make_vlp_transform(
            image_columns={"image": "image"},
            text_column="caption",
            multi_caption=True,
            seed=seed,
        )
    )
    return ds


def load_vqav2(
    split: str,
    hub_id: str = "lmms-lab/VQAv2",
    seed: Optional[int] = None,
):
    """VQAv2 from `lmms-lab/VQAv2`.

    Splits: ``validation``, ``test``, ``testdev``. Note: `lmms-lab/VQAv2`
    does not host the train split — for pretraining you still need the
    original VQAv2 train set from another source.

    Schema: ``image`` (PIL), ``question`` (str), ``question_id`` (int),
    ``multiple_choice_answer`` (str), ``answers`` (list of 10 annotations),
    ``image_id`` (int).
    """
    if split == "val":
        split = "validation"
    if split not in ("validation", "test", "testdev"):
        raise ValueError(
            f"split must be one of validation/test/testdev, got {split!r}"
        )

    ds = load_dataset(hub_id, split=split)
    ds = ds.cast_column("image", DSImage(decode=True))
    ds = ds.with_transform(
        make_vlp_transform(
            image_columns={"image": "image"},
            text_column="question",
            multi_caption=False,
            seed=seed,
        )
    )
    return ds


def load_nlvr2(
    split: str,
    hub_id: str = "lmms-lab/NLVR2",
    seed: Optional[int] = None,
):
    """NLVR2 from `lmms-lab/NLVR2`.

    Splits: ``train``, ``dev``, ``test``.

    Schema: ``sentence`` (str), ``left_image`` (PIL), ``right_image`` (PIL),
    ``label`` (str or int).

    Output schema after transform: ``image_0``, ``image_1``, ``text``,
    ``label``. Pair with ``VLPCollator(image_keys=("image_0", "image_1"))``.
    """
    if split == "val":
        split = "dev"
    if split not in ("train", "dev", "test"):
        raise ValueError(f"split must be one of train/dev/test, got {split!r}")

    ds = load_dataset(hub_id, split=split)
    ds = ds.cast_column("left_image", DSImage(decode=True))
    ds = ds.cast_column("right_image", DSImage(decode=True))
    ds = ds.with_transform(
        make_vlp_transform(
            image_columns={"left_image": "image_0", "right_image": "image_1"},
            text_column="sentence",
            multi_caption=False,
            seed=seed,
        )
    )
    return ds


def _load_refcoco_family(
    hub_id: str,
    split: str,
    allowed_splits,
    seed: Optional[int] = None,
):
    if split not in allowed_splits:
        raise ValueError(
            f"split must be one of {sorted(allowed_splits)}, got {split!r}"
        )
    ds = load_dataset(hub_id, split=split)
    ds = ds.cast_column("image", DSImage(decode=True))
    ds = ds.with_transform(
        make_vlp_transform(
            image_columns={"image": "image"},
            text_column="question",
            multi_caption=False,
            seed=seed,
        )
    )
    return ds


def load_refcoco(split, seed=None):
    """RefCOCO from `lmms-lab/RefCOCO`. Splits: val, test, testA, testB.

    Schema: ``image``, ``question`` (referring expression), ``answer`` (list),
    ``bbox`` ([x, y, w, h]), ``segmentation``, ``file_name``.
    """
    return _load_refcoco_family(
        "lmms-lab/RefCOCO", split,
        allowed_splits={"val", "test", "testA", "testB"},
        seed=seed,
    )


def load_refcocoplus(split, seed=None):
    """RefCOCO+ from `lmms-lab/RefCOCOplus`. Splits: val, testA, testB."""
    return _load_refcoco_family(
        "lmms-lab/RefCOCOplus", split,
        allowed_splits={"val", "testA", "testB"},
        seed=seed,
    )


def load_refcocog(split, seed=None):
    """RefCOCOg from `lmms-lab/RefCOCOg`. Splits: val, test."""
    return _load_refcoco_family(
        "lmms-lab/RefCOCOg", split,
        allowed_splits={"val", "test"},
        seed=seed,
    )


def _load_wds_captioning(
    hub_id: str,
    split: str,
    streaming: bool,
    seed: Optional[int],
):
    """Shared loader for WebDataset-format caption datasets (`pixparse/cc3m-wds`,
    `pixparse/cc12m-wds`).

    Schema in the source: ``{"__key__": str, "jpg": bytes, "txt": str}``.
    Output schema after transform: ``{"image": PIL.Image, "text": str}``;
    the collator handles the PIL→Tensor conversion at batch time.
    """
    ds = load_dataset(hub_id, split=split, streaming=streaming)
    transform = make_vlp_transform(
        image_columns={"jpg": "image"},
        text_column="txt",
        multi_caption=False,
        seed=seed,
    )
    if streaming:
        # IterableDataset doesn't support `.with_transform`; use `.map` instead.
        # Drop ALL source columns from the output — even though the transform
        # consumes jpg/txt and re-emits image/text, leaving any input column
        # in the output trips up feature encoding downstream.
        ds = ds.map(
            transform, batched=True,
            remove_columns=["__key__", "jpg", "txt"],
            features=_WDS_OUT_FEATURES,
        )
    else:
        ds = ds.with_transform(transform)
    return ds


def load_cc3m(
    split: str = "train",
    hub_id: str = "pixparse/cc3m-wds",
    streaming: bool = True,
    seed: Optional[int] = None,
):
    """CC3M from `pixparse/cc3m-wds` (WebDataset format, ~281 GB).

    Streaming is the default and the realistic mode — CC3M is meant to be
    iterated, not random-accessed. Pass ``streaming=False`` only if you have
    a local pre-downloaded mirror.

    Splits: ``train`` (~2.9 M), ``val`` (~13 K).

    Caveat: the `pixparse` mirror redistributes CC3M images, which sits in a
    legal grey zone with Google's original terms. Fine for research; check
    before publishing checkpoints trained on it.
    """
    if split not in ("train", "val", "validation"):
        raise ValueError(f"split must be one of train/val, got {split!r}")
    if split == "val":
        split = "validation"
    return _load_wds_captioning(hub_id, split, streaming, seed)


def load_cc12m(
    split: str = "train",
    hub_id: str = "pixparse/cc12m-wds",
    streaming: bool = True,
    seed: Optional[int] = None,
):
    """CC12M from `pixparse/cc12m-wds` (WebDataset format).

    Single ``train`` split. Same WDS schema and streaming considerations as
    CC3M.
    """
    if split != "train":
        raise ValueError(f"split must be one of train, got {split!r}")
    return _load_wds_captioning(hub_id, split, streaming, seed)


def load_coco_karpathy(
    split: str,
    hub_id: str = "namkha1032/coco-karpathy",
    seed: Optional[int] = None,
):
    """COCO 2014 captions with Karpathy splits from `namkha1032/coco-karpathy`.

    Splits: ``train`` (113 K, restval folded in), ``val`` (5 K), ``test``
    (5 K). Schema in the source: ``image_id`` (str), ``image`` (PIL),
    ``image_width (px)`` (int — dropped on load), ``captions`` (list of 5-7
    strings).

    Note: this is a community re-upload with unspecified license metadata.
    COCO 2014's underlying license is CC-BY-4.0 so functionally it's
    permissive, but flag this when publishing checkpoint provenance.
    """
    if split not in ("train", "val", "validation", "test"):
        raise ValueError(
            f"split must be one of train/val/test, got {split!r}"
        )
    if split == "val":
        split = "validation"

    ds = load_dataset(hub_id, split=split)
    # The `image_width (px)` column has an awkward name and isn't used.
    if "image_width (px)" in ds.column_names:
        ds = ds.remove_columns(["image_width (px)"])
    ds = ds.cast_column("image", DSImage(decode=True))
    ds = ds.with_transform(
        make_vlp_transform(
            image_columns={"image": "image"},
            text_column="captions",
            multi_caption=True,
            seed=seed,
        )
    )
    return ds


def load_visual_genome(
    split: str = "train",
    config: str = "region_descriptions_v1.2.0",
    hub_id: str = "ranjaykrishna/visual_genome",
    seed: Optional[int] = None,
):
    """Visual Genome region descriptions from `ranjaykrishna/visual_genome`.

    Single ``train`` split (~108 K images). The ``regions`` column is a list
    of per-image bounding-box descriptions; we extract ``regions[].phrase``
    into a ``caption`` column and feed that through the multi-caption
    transform (one phrase sampled per row per epoch).

    Other configs available (``objects_v1.2.0``, ``attributes_v1.2.0``,
    ``relationships_v1.2.0``, ``question_answers_v1.2.0``) — pass via
    ``config``.
    """
    if split != "train":
        raise ValueError(
            f"split must be one of train (the only split available), got {split!r}"
        )

    ds = load_dataset(hub_id, config, split=split)

    def _extract_phrases(batch):
        # `Sequence(struct)` features are exposed as struct-of-arrays
        # (a dict of lists) in batched .map. `[struct]` features are
        # exposed as list-of-dicts. Handle both.
        captions = []
        for regions in batch["regions"]:
            if isinstance(regions, dict):
                captions.append(list(regions["phrase"]))
            else:
                captions.append([r["phrase"] for r in regions])
        return {"caption": captions}

    ds = ds.map(
        _extract_phrases,
        batched=True,
        remove_columns=["regions"],
    )
    ds = ds.cast_column("image", DSImage(decode=True))
    ds = ds.with_transform(
        make_vlp_transform(
            image_columns={"image": "image"},
            text_column="caption",
            multi_caption=True,
            seed=seed,
        )
    )
    return ds


def load_sbu(
    path: str,
    seed: Optional[int] = None,
):
    """SBU Captions from a local img2dataset WebDataset mirror.

    No usable Hub mirror exists: `vicenteor/sbu_captions` is URL-only and
    most Flickr URLs are dead. Pre-download with img2dataset:

        img2dataset --url_list sbu_urls.tsv --input_format tsv \\
                    --url_col 0 --caption_col 1 \\
                    --output_folder sbu_wds --output_format webdataset \\
                    --image_size 256 --processes_count 16 --thread_count 32

    Then pass the output directory as ``path``. Loads via streaming WDS so
    the dataset behaves like CC3M downstream.
    """
    import glob

    tars = sorted(glob.glob(f"{path}/*.tar"))
    if not tars:
        raise FileNotFoundError(
            f"No .tar WebDataset shards found in {path!r}. "
            "Run img2dataset against the SBU URL list first."
        )
    ds = load_dataset(
        "webdataset", data_files={"train": tars}, split="train", streaming=True
    )
    transform = make_vlp_transform(
        image_columns={"jpg": "image"},
        text_column="txt",
        multi_caption=False,
        seed=seed,
    )
    ds = ds.map(
        transform, batched=True,
        remove_columns=["__key__", "jpg", "txt"],
        features=_WDS_OUT_FEATURES,
    )
    return ds


def load_snli_ve(
    split: str,
    hub_id: str = "HuggingFaceM4/SNLI-VE",
    seed: Optional[int] = None,
):
    """SNLI-VE from `HuggingFaceM4/SNLI-VE`.

    Splits: ``train``, ``val``, ``test``.

    Requires ``trust_remote_code=True`` (custom loading script) and a local
    copy of ``flickr30k-images.tar.gz`` (the script joins SNLI text labels
    with locally-stored Flickr30k images). See the dataset card for archive
    download instructions.

    Schema in source: ``image`` (PIL), ``filename`` (str), ``premise`` (str),
    ``hypothesis`` (str), ``label`` (ClassLabel: entailment/neutral/
    contradiction). The model's text input is ``hypothesis`` — the premise
    is the image, not a text input.

    Known limitation: ~9.2% of the neutral-class labels in the original
    SNLI-VE paper are noisy, per the published authors.
    """
    if split == "val":
        split = "validation"
    if split not in ("train", "validation", "test"):
        raise ValueError(f"split must be one of train/val/test, got {split!r}")

    ds = load_dataset(hub_id, split=split, trust_remote_code=True)
    ds = ds.cast_column("image", DSImage(decode=True))
    ds = ds.with_transform(
        make_vlp_transform(
            image_columns={"image": "image"},
            text_column="hypothesis",
            multi_caption=False,
            seed=seed,
        )
    )
    return ds


# --- GLUE -------------------------------------------------------------------

_GLUE_COLUMNS = {
    "mrpc": ("sentence1", "sentence2"),
    "rte": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "stsb": ("sentence1", "sentence2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "sst2": ("sentence", None),
    "cola": ("sentence", None),
}


def load_glue(
    task: str,
    split: str,
    hub_id: str = "nyu-mll/glue",
    seed: Optional[int] = None,
):
    """GLUE benchmark from `nyu-mll/glue` — text-only, no image.

    ``task`` is one of: mrpc, rte, wnli, stsb, mnli, qnli, qqp, sst2, cola.
    ``split`` is one of: train, validation, test. Test labels are hidden in
    GLUE; use ``validation`` for evaluation.

    Output schema after transform: ``{"text": str, "text_pair"?: str,
    "label": int, "idx"?: int}``. For paired-sentence tasks, source columns
    are renamed to ``text`` / ``text_pair`` for clean tokenization via the
    ``tokenizer(text, text_pair=...)`` API.

    Pair with ``VLPCollator(image_keys=())`` to skip image processing.

    Note for MNLI: ``validation`` is treated as ``validation_matched``.
    Use ``validation_mismatched`` explicitly for the mismatched split.
    """
    if task not in _GLUE_COLUMNS:
        raise ValueError(
            f"task must be one of {sorted(_GLUE_COLUMNS)}, got {task!r}"
        )
    valid_splits = ("train", "validation", "test")
    if task == "mnli":
        valid_splits = (
            "train",
            "validation",
            "validation_matched",
            "validation_mismatched",
            "test_matched",
            "test_mismatched",
        )
    if split not in valid_splits:
        raise ValueError(
            f"split must be one of {valid_splits}, got {split!r}"
        )
    if task == "mnli" and split == "validation":
        split = "validation_matched"

    col_a, col_b = _GLUE_COLUMNS[task]
    ds = load_dataset(hub_id, task, split=split)

    def transform(batch):
        out = {"text": list(batch[col_a])}
        if col_b is not None:
            out["text_pair"] = list(batch[col_b])
        if "label" in batch:
            out["label"] = list(batch["label"])
        if "idx" in batch:
            out["idx"] = list(batch["idx"])
        return out

    ds = ds.with_transform(transform)
    return ds
