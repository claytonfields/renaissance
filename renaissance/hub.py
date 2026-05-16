"""
HuggingFace Hub integration for RenaissanceTransformer.

RenaissanceHubConfig — PretrainedConfig subclass that serializes all model
hyperparameters (ModelConfig + TaskConfig fields) to config.json.

push_to_hub — thin wrapper around HfApi.upload_folder.
"""

import os
from typing import Any, Dict, List, Optional

from transformers import PretrainedConfig

from .config_schema import ALL_LOSS_NAMES, ModelConfig, TaskConfig
from dataclasses import fields as dc_fields


def _default_loss_names() -> Dict[str, int]:
    return dict(ALL_LOSS_NAMES)


class RenaissanceHubConfig(PretrainedConfig):
    model_type = "renaissance"

    def __init__(
        self,
        # architecture selector (stored as 'arch' to avoid clashing with
        # PretrainedConfig.model_type which is the HF registry key)
        arch: str = "two-tower",
        # --- one-tower ---
        encoder: str = "google/electra-small-discriminator",
        pooler_type: str = "double",
        random_init_encoder: bool = False,
        encoder_manual_configuration: bool = False,
        hidden_size: int = 192,
        num_heads: int = 4,
        num_layers: int = 12,
        mlp_ratio: int = 4,
        drop_rate: float = 0.1,
        embedding_size: int = 96,
        # --- two-tower image ---
        image_encoder: str = "facebook/deit-tiny-patch16-224",
        random_init_vision_encoder: bool = False,
        image_encoder_manual_configuration: bool = False,
        image_encoder_hidden_size: int = 192,
        image_encoder_num_heads: int = 4,
        image_encoder_num_layers: int = 12,
        image_encoder_mlp_ratio: int = 4,
        image_encoder_drop_rate: float = 0.1,
        image_encoder_embedding_size: int = 128,
        image_size: int = 224,
        original_image_size: int = 224,
        patch_size: int = 16,
        image_only: bool = False,
        # --- two-tower text ---
        text_encoder: str = "google/electra-small-discriminator",
        random_init_text_encoder: bool = False,
        text_encoder_manual_configuration: bool = False,
        text_encoder_hidden_size: int = 192,
        text_encoder_num_heads: int = 4,
        text_encoder_num_layers: int = 12,
        text_encoder_mlp_ratio: int = 4,
        text_encoder_drop_rate: float = 0.1,
        text_encoder_embedding_size: int = 64,
        max_text_len: int = 40,
        vocab_size: int = 30522,
        # --- cross-modal fusion ---
        cross_layer_hidden_size: int = 256,
        num_cross_layers: int = 6,
        num_cross_layer_heads: int = 4,
        cross_layer_mlp_ratio: int = 4,
        cross_layer_drop_rate: float = 0.1,
        # --- freeze flags ---
        freeze_image_encoder: bool = False,
        freeze_text_encoder: bool = False,
        freeze_cross_modal_layers: bool = False,
        # --- task ---
        tasks: Optional[List[str]] = None,
        loss_names: Optional[Dict[str, int]] = None,
        task_config: Optional[Dict[str, Any]] = None,
        whole_word_masking: bool = False,
        mlm_prob: float = 0.15,
        draw_false_image: int = 1,
        draw_false_text: int = 0,
        get_recall_metric: bool = False,
        vqav2_label_size: int = 3129,
        max_bb: int = 20,
        ref_res_head_layers: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.arch = arch
        self.encoder = encoder
        self.pooler_type = pooler_type
        self.random_init_encoder = random_init_encoder
        self.encoder_manual_configuration = encoder_manual_configuration
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.embedding_size = embedding_size
        self.image_encoder = image_encoder
        self.random_init_vision_encoder = random_init_vision_encoder
        self.image_encoder_manual_configuration = image_encoder_manual_configuration
        self.image_encoder_hidden_size = image_encoder_hidden_size
        self.image_encoder_num_heads = image_encoder_num_heads
        self.image_encoder_num_layers = image_encoder_num_layers
        self.image_encoder_mlp_ratio = image_encoder_mlp_ratio
        self.image_encoder_drop_rate = image_encoder_drop_rate
        self.image_encoder_embedding_size = image_encoder_embedding_size
        self.image_size = image_size
        self.original_image_size = original_image_size
        self.patch_size = patch_size
        self.image_only = image_only
        self.text_encoder = text_encoder
        self.random_init_text_encoder = random_init_text_encoder
        self.text_encoder_manual_configuration = text_encoder_manual_configuration
        self.text_encoder_hidden_size = text_encoder_hidden_size
        self.text_encoder_num_heads = text_encoder_num_heads
        self.text_encoder_num_layers = text_encoder_num_layers
        self.text_encoder_mlp_ratio = text_encoder_mlp_ratio
        self.text_encoder_drop_rate = text_encoder_drop_rate
        self.text_encoder_embedding_size = text_encoder_embedding_size
        self.max_text_len = max_text_len
        self.vocab_size = vocab_size
        self.cross_layer_hidden_size = cross_layer_hidden_size
        self.num_cross_layers = num_cross_layers
        self.num_cross_layer_heads = num_cross_layer_heads
        self.cross_layer_mlp_ratio = cross_layer_mlp_ratio
        self.cross_layer_drop_rate = cross_layer_drop_rate
        self.freeze_image_encoder = freeze_image_encoder
        self.freeze_text_encoder = freeze_text_encoder
        self.freeze_cross_modal_layers = freeze_cross_modal_layers
        self.tasks = tasks if tasks is not None else []
        self.loss_names = loss_names if loss_names is not None else _default_loss_names()
        self.task_config = task_config if task_config is not None else {}
        self.whole_word_masking = whole_word_masking
        self.mlm_prob = mlm_prob
        self.draw_false_image = draw_false_image
        self.draw_false_text = draw_false_text
        self.get_recall_metric = get_recall_metric
        self.vqav2_label_size = vqav2_label_size
        self.max_bb = max_bb
        self.ref_res_head_layers = ref_res_head_layers

    @classmethod
    def from_flat_config(cls, flat: Dict[str, Any]) -> "RenaissanceHubConfig":
        """Build a RenaissanceHubConfig from the flat dict RenaissanceTransformer uses."""
        model_fields = {f.name for f in dc_fields(ModelConfig)}
        task_fields = {f.name for f in dc_fields(TaskConfig)}
        kwargs: Dict[str, Any] = {}
        for k, v in flat.items():
            if k == "model_type":
                kwargs["arch"] = v
            elif k in model_fields or k in task_fields:
                kwargs[k] = v
        return cls(**kwargs)

    def to_flat_config(self) -> Dict[str, Any]:
        """Return the flat dict that RenaissanceTransformer expects."""
        d: Dict[str, Any] = {
            "model_type": self.arch,
            "encoder": self.encoder,
            "pooler_type": self.pooler_type,
            "random_init_encoder": self.random_init_encoder,
            "encoder_manual_configuration": self.encoder_manual_configuration,
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "mlp_ratio": self.mlp_ratio,
            "drop_rate": self.drop_rate,
            "embedding_size": self.embedding_size,
            "image_encoder": self.image_encoder,
            "random_init_vision_encoder": self.random_init_vision_encoder,
            "image_encoder_manual_configuration": self.image_encoder_manual_configuration,
            "image_encoder_hidden_size": self.image_encoder_hidden_size,
            "image_encoder_num_heads": self.image_encoder_num_heads,
            "image_encoder_num_layers": self.image_encoder_num_layers,
            "image_encoder_mlp_ratio": self.image_encoder_mlp_ratio,
            "image_encoder_drop_rate": self.image_encoder_drop_rate,
            "image_encoder_embedding_size": self.image_encoder_embedding_size,
            "image_size": self.image_size,
            "original_image_size": self.original_image_size,
            "patch_size": self.patch_size,
            "image_only": self.image_only,
            "text_encoder": self.text_encoder,
            "random_init_text_encoder": self.random_init_text_encoder,
            "text_encoder_manual_configuration": self.text_encoder_manual_configuration,
            "text_encoder_hidden_size": self.text_encoder_hidden_size,
            "text_encoder_num_heads": self.text_encoder_num_heads,
            "text_encoder_num_layers": self.text_encoder_num_layers,
            "text_encoder_mlp_ratio": self.text_encoder_mlp_ratio,
            "text_encoder_drop_rate": self.text_encoder_drop_rate,
            "text_encoder_embedding_size": self.text_encoder_embedding_size,
            "max_text_len": self.max_text_len,
            "vocab_size": self.vocab_size,
            "cross_layer_hidden_size": self.cross_layer_hidden_size,
            "num_cross_layers": self.num_cross_layers,
            "num_cross_layer_heads": self.num_cross_layer_heads,
            "cross_layer_mlp_ratio": self.cross_layer_mlp_ratio,
            "cross_layer_drop_rate": self.cross_layer_drop_rate,
            "freeze_image_encoder": self.freeze_image_encoder,
            "freeze_text_encoder": self.freeze_text_encoder,
            "freeze_cross_modal_layers": self.freeze_cross_modal_layers,
            "tasks": self.tasks,
            "loss_names": self.loss_names,
            "task_config": self.task_config,
            "whole_word_masking": self.whole_word_masking,
            "mlm_prob": self.mlm_prob,
            "draw_false_image": self.draw_false_image,
            "draw_false_text": self.draw_false_text,
            "get_recall_metric": self.get_recall_metric,
            "vqav2_label_size": self.vqav2_label_size,
            "max_bb": self.max_bb,
            "ref_res_head_layers": self.ref_res_head_layers,
            # experiment defaults — not serialized in hub config, but required
            # by RenaissanceTransformer.__init__ to skip legacy checkpoint loading
            "load_path": "",
            "test_only": False,
            "log_dir": "result",
            "seed": 0,
            "exp_name": "renaissance",
            "resume_from": None,
        }
        return d


def push_to_hub(model, repo_id: str, token: Optional[str] = None) -> str:
    """Save model to a temp dir and upload the whole folder to the Hub."""
    import tempfile
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=token)
    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        url = api.upload_folder(folder_path=tmp, repo_id=repo_id, repo_type="model")
    return url
