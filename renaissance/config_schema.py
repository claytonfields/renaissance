"""
Typed configuration schema for Renaissance.

Each dataclass group maps to a top-level YAML key.  The top-level
RenaissanceConfig is a flat dict when passed to RenaissanceModel —
use config_schema.to_flat_dict(cfg) to produce it.

CLI usage (via run.py):
    python run.py config.yaml key=value key2=value2
"""

from __future__ import annotations
from dataclasses import dataclass, field, fields, asdict
from typing import List, Optional, Dict, Any


# ---------------------------------------------------------------------------
# Loss registry
# ---------------------------------------------------------------------------

ALL_LOSS_NAMES: Dict[str, int] = {
    "itm": 0, "mlm": 0, "mpp": 0, "vqa": 0, "vcr": 0, "vcr_qar": 0,
    "nlvr2": 0, "irtr": 0, "contras": 0, "snli": 0, "ref": 0, "ref2": 0,
    "mrpc": 0, "rte": 0, "wnli": 0, "sst2": 0, "qqp": 0, "qnli": 0,
    "mnli": 0, "cola": 0, "cifar10": 0,
}


def _loss_names(overrides: Dict[str, int]) -> Dict[str, int]:
    ret = dict(ALL_LOSS_NAMES)
    ret.update(overrides)
    return ret


# ---------------------------------------------------------------------------
# Config groups
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    model_type: str = "two-tower"

    # One-tower
    encoder: str = "google/electra-small-discriminator"
    pooler_type: str = "double"
    random_init_encoder: bool = False
    encoder_manual_configuration: bool = False
    hidden_size: int = 192
    num_heads: int = 4
    num_layers: int = 12
    mlp_ratio: int = 4
    drop_rate: float = 0.1
    embedding_size: int = 96

    # Two-tower image
    image_encoder: str = "facebook/deit-tiny-patch16-224"
    random_init_vision_encoder: bool = False
    image_encoder_manual_configuration: bool = False
    image_encoder_hidden_size: int = 192
    image_encoder_num_heads: int = 4
    image_encoder_num_layers: int = 12
    image_encoder_mlp_ratio: int = 4
    image_encoder_drop_rate: float = 0.1
    image_encoder_embedding_size: int = 128
    image_size: int = 224
    original_image_size: int = 224
    patch_size: int = 16
    image_only: bool = False

    # Two-tower text
    text_encoder: str = "google/electra-small-discriminator"
    random_init_text_encoder: bool = False
    text_encoder_manual_configuration: bool = False
    text_encoder_hidden_size: int = 192
    text_encoder_num_heads: int = 4
    text_encoder_num_layers: int = 12
    text_encoder_mlp_ratio: int = 4
    text_encoder_drop_rate: float = 0.1
    text_encoder_embedding_size: int = 64
    max_text_len: int = 40
    vocab_size: int = 30522

    # Cross-modal fusion
    cross_layer_hidden_size: int = 256
    num_cross_layers: int = 6
    num_cross_layer_heads: int = 4
    cross_layer_mlp_ratio: int = 4
    cross_layer_drop_rate: float = 0.1

    # Freeze flags
    freeze_image_encoder: bool = False
    freeze_text_encoder: bool = False
    freeze_cross_modal_layers: bool = False


@dataclass
class TaskConfig:
    """Which objectives are active.

    Two equivalent ways to declare the active set:

    - ``tasks``: an explicit list, e.g. ``["mlm", "itm"]`` (preferred).
    - ``loss_names``: the legacy ``{name: 0|1}`` dict.

    Use either; `normalize_tasks` reconciles them so both are always
    present and consistent downstream. If ``tasks`` is non-empty it is
    authoritative and ``loss_names`` is rebuilt from it; otherwise
    ``tasks`` is derived from the ``loss_names`` entries that are > 0.
    """
    tasks: List[str] = field(default_factory=list)
    loss_names: Dict[str, int] = field(default_factory=lambda: _loss_names({"itm": 1, "mlm": 1}))
    # Optional per-task overrides, e.g. {"vqa": {"label_size": 3129}}.
    task_config: Dict[str, Any] = field(default_factory=dict)

    # Task-specific settings
    whole_word_masking: bool = False
    mlm_prob: float = 0.15
    draw_false_image: int = 1
    draw_false_text: int = 0
    get_recall_metric: bool = False
    vqav2_label_size: int = 3129
    max_bb: int = 20
    ref_res_head_layers: int = 2


@dataclass
class DataConfig:
    datasets: List[str] = field(default_factory=lambda: ["coco", "vg"])
    data_root: str = "data/arrow/"
    train_transform_keys: List[str] = field(default_factory=lambda: ["imagenet"])
    val_transform_keys: List[str] = field(default_factory=lambda: ["imagenet"])
    batch_size: int = 256
    per_gpu_batchsize: int = 32
    eval_batch_size: int = 32
    num_workers: int = 12

    # "modern" → renaissance/data (HF Hub-first). Default since the 1.3 line.
    # "legacy" → renaissance/datamodules + renaissance/datasets (pyarrow IPC,
    #   Lightning-coupled). Deprecated; kept for reading pre-serialized Arrow
    #   data on disk until the legacy data layer is removed.
    backend: str = "modern"
    # Per-dataset extra kwargs for the modern runner (e.g. {"sbu": {"path": "..."},
    # "glue": {"task": "mrpc"}, "vg": {"config": "region_descriptions_v1.2.0"}}).
    dataset_kwargs: Dict[str, Any] = field(default_factory=dict)
    # Multi-dataset interleave (modern backend only). When `datasets` has more
    # than one entry, the runner uses `datasets.interleave_datasets` with these
    # weights and the given stopping strategy.
    dataset_probs: Optional[List[float]] = None
    stopping_strategy: str = "first_exhausted"


@dataclass
class TrainingConfig:
    optim_type: str = "adamw"
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    decay_power: Any = 1            # int or "cosine"
    max_epoch: int = 100
    max_steps: int = 100_000
    warmup_steps: Any = 10_000     # int or float fraction
    end_lr: float = 0.0
    lr_mult_head: float = 5.0
    lr_mult_cross_modal: float = 5.0
    precision: Any = 32            # 32, 16, or "bf16"
    num_gpus: Any = 1              # int or list of ints
    num_nodes: int = 1
    val_check_interval: float = 1.0
    fast_dev_run: bool = False


@dataclass
class ExperimentConfig:
    exp_name: str = "renaissance"
    seed: int = 0
    load_path: str = ""
    resume_from: Optional[str] = None
    test_only: bool = False
    log_dir: str = "result"


@dataclass
class RenaissanceConfig:
    """Top-level config — compose all groups."""
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def normalize_tasks(flat: Dict[str, Any]) -> Dict[str, Any]:
    """Reconcile ``tasks`` (list) and ``loss_names`` (dict) in a flat config.

    After this runs, both keys are present and consistent:

    - If ``tasks`` is non-empty it is authoritative — ``loss_names`` is
      rebuilt as ``ALL_LOSS_NAMES`` with those names set to 1. (So a
      ``tasks`` override wins over any inherited ``loss_names``.)
    - Else if ``loss_names`` has entries > 0, ``tasks`` is derived from
      them (in ``ALL_LOSS_NAMES`` order).
    - Else both stay at their defaults.

    Unknown task names raise ``ValueError``. Idempotent. Mutates and
    returns ``flat``.
    """
    tasks = flat.get("tasks") or []
    loss_names = flat.get("loss_names") or {}

    if tasks:
        unknown = [t for t in tasks if t not in ALL_LOSS_NAMES]
        if unknown:
            raise ValueError(
                f"Unknown task name(s) {unknown}; valid: {sorted(ALL_LOSS_NAMES)}"
            )
        flat["loss_names"] = _loss_names({t: 1 for t in tasks})
        flat["tasks"] = [t for t in ALL_LOSS_NAMES if t in set(tasks)]
    else:
        flat["loss_names"] = _loss_names(loss_names)
        flat["tasks"] = [
            name for name, weight in flat["loss_names"].items()
            if weight and weight > 0
        ]
    return flat


def to_flat_dict(cfg: RenaissanceConfig) -> Dict[str, Any]:
    """Flatten RenaissanceConfig into the dict RenaissanceModel expects."""
    d: Dict[str, Any] = {}
    d.update(asdict(cfg.experiment))
    d.update(asdict(cfg.model))
    d.update(asdict(cfg.task))
    d.update(asdict(cfg.data))
    d.update(asdict(cfg.training))
    return normalize_tasks(d)


def from_omegaconf(omega_cfg) -> Dict[str, Any]:
    """Convert an OmegaConf DictConfig (loaded from YAML + CLI overrides)
    to the flat dict that RenaissanceModel expects.

    The YAML can be either:
    - Grouped (top-level keys: experiment, model, task, data, training)
    - Flat (all keys at top level, for simple single-file configs)
    """
    from omegaconf import OmegaConf

    raw = OmegaConf.to_container(omega_cfg, resolve=True)

    # If the YAML uses grouped structure, flatten it
    if any(k in raw for k in ("experiment", "model", "task", "data", "training")):
        flat: Dict[str, Any] = {}
        for group in ("experiment", "model", "task", "data", "training"):
            if group in raw:
                flat.update(raw[group])
        # Allow top-level overrides to win over group values
        for k, v in raw.items():
            if k not in ("experiment", "model", "task", "data", "training"):
                flat[k] = v
        return normalize_tasks(flat)

    # Already flat
    return normalize_tasks(raw)
