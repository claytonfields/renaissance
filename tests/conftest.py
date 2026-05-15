import pytest
import torch

# ---------------------------------------------------------------------------
# Batch constants — kept tiny so tests run fast on CPU
# ---------------------------------------------------------------------------
BS = 2
ONE_TOWER_IMG = 32    # OneTowerEncoder builds ViTEmbeddings from scratch, any size works
TWO_TOWER_IMG = 224   # DeiT position embeddings are fixed to 224 by the HF config
PATCH = 16
TEXT_LEN = 16
VOCAB = 30522
CROSS_HIDDEN = 64     # small cross-modal hidden size
NUM_CROSS = 2         # minimal cross-modal depth
MAX_BB = 4            # bounding-box regions per image for ref task

# Mirrors _loss_names() in config.py — all tasks off by default
ALL_LOSS_NAMES = {
    "itm": 0, "mlm": 0, "mpp": 0, "vqa": 0, "vcr": 0, "vcr_qar": 0,
    "nlvr2": 0, "irtr": 0, "contras": 0, "snli": 0, "ref": 0, "ref2": 0,
    "mrpc": 0, "rte": 0, "wnli": 0, "sst2": 0, "qqp": 0, "qnli": 0,
    "mnli": 0, "cola": 0, "cifar10": 0,
}


# ---------------------------------------------------------------------------
# Base config shared by both architectures
# ---------------------------------------------------------------------------
def _base_config():
    return {
        # Experiment
        "exp_name": "smoke_test",
        "load_path": "",
        "test_only": False,
        "fast_dev_run": False,
        "get_recall_metric": False,
        # Batch / data
        "per_gpu_batchsize": BS,
        "batch_size": BS,
        "datasets": ["coco"],
        "draw_false_image": 1,
        "draw_false_text": 0,
        # Text
        "vocab_size": VOCAB,
        "max_text_len": TEXT_LEN,
        # Image
        "image_size": ONE_TOWER_IMG,
        "original_image_size": ONE_TOWER_IMG,
        "patch_size": PATCH,
        "image_only": False,
        # Loss names (all off — each fixture enables what it needs)
        "loss_names": dict(ALL_LOSS_NAMES),
        "vqav2_label_size": 10,
        "max_bb": MAX_BB,
        "ref_res_head_layers": 2,
        # Optimizer (needed by configure_optimizers / set_schedule)
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "lr_mult_head": 5,
        "lr_mult_cross_modal": 5,
        "end_lr": 0,
        "decay_power": 1,
        "optim_type": "adamw",
        "warmup_steps": 10,
        "max_epoch": 1,
        "max_steps": 10,
        # Lightning trainer (not used directly by model but stored in hparams)
        "precision": 32,
        "val_check_interval": 1.0,
        # Two-tower cross-modal
        "cross_layer_hidden_size": CROSS_HIDDEN,
        "num_cross_layers": NUM_CROSS,
        "num_cross_layer_heads": 4,
        "cross_layer_mlp_ratio": 4,
        "cross_layer_drop_rate": 0.0,
        # Two-tower encoder flags
        "image_encoder": "facebook/deit-tiny-patch16-224",
        "text_encoder": "google/electra-small-discriminator",
        "random_init_vision_encoder": True,
        "random_init_text_encoder": True,
        "image_encoder_manual_configuration": False,
        "text_encoder_manual_configuration": False,
        "freeze_image_encoder": False,
        "freeze_text_encoder": False,
        "freeze_cross_modal_layers": False,
        # Two-tower manual dim overrides (only active if *_manual_configuration=True)
        "image_encoder_hidden_size": 192,   # DeiT-tiny actual hidden size
        "image_encoder_num_heads": 3,
        "image_encoder_num_layers": 12,
        "image_encoder_mlp_ratio": 4,
        "image_encoder_drop_rate": 0.0,
        "image_encoder_embedding_size": 128,
        "text_encoder_hidden_size": 256,    # electra-small actual hidden size
        "text_encoder_num_heads": 4,
        "text_encoder_num_layers": 12,
        "text_encoder_mlp_ratio": 4,
        "text_encoder_drop_rate": 0.0,
        "text_encoder_embedding_size": 128,
        # One-tower flags
        "encoder": "google/electra-small-discriminator",
        "random_init_encoder": True,
        "encoder_manual_configuration": False,
        "pooler_type": "double",
        "drop_rate": 0.0,
        # One-tower manual dim overrides
        "hidden_size": 256,
        "num_heads": 4,
        "num_layers": 6,
        "mlp_ratio": 4,
        "embedding_size": 128,
    }


# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def one_tower_pretrain_config():
    cfg = _base_config()
    cfg.update({
        "model_type": "one-tower",
        "image_size": ONE_TOWER_IMG,
        "original_image_size": ONE_TOWER_IMG,
        "loss_names": {**ALL_LOSS_NAMES, "mlm": 1, "itm": 1},
    })
    return cfg


@pytest.fixture(scope="session")
def two_tower_pretrain_config():
    cfg = _base_config()
    cfg.update({
        "model_type": "two-tower",
        "image_size": TWO_TOWER_IMG,
        "original_image_size": TWO_TOWER_IMG,
        "loss_names": {**ALL_LOSS_NAMES, "mlm": 1, "itm": 1},
    })
    return cfg


@pytest.fixture(scope="session")
def two_tower_snli_config():
    cfg = _base_config()
    cfg.update({
        "model_type": "two-tower",
        "image_size": TWO_TOWER_IMG,
        "original_image_size": TWO_TOWER_IMG,
        "loss_names": {**ALL_LOSS_NAMES, "snli": 1},
    })
    return cfg


@pytest.fixture(scope="session")
def two_tower_nlvr2_config():
    cfg = _base_config()
    cfg.update({
        "model_type": "two-tower",
        "image_size": TWO_TOWER_IMG,
        "original_image_size": TWO_TOWER_IMG,
        "loss_names": {**ALL_LOSS_NAMES, "nlvr2": 1},
    })
    return cfg


@pytest.fixture(scope="session")
def two_tower_ref_config():
    cfg = _base_config()
    cfg.update({
        "model_type": "two-tower",
        "image_size": TWO_TOWER_IMG,
        "original_image_size": TWO_TOWER_IMG,
        "loss_names": {**ALL_LOSS_NAMES, "ref": 1},
    })
    return cfg


@pytest.fixture(scope="session")
def two_tower_vqa_config():
    cfg = _base_config()
    cfg.update({
        "model_type": "two-tower",
        "image_size": TWO_TOWER_IMG,
        "original_image_size": TWO_TOWER_IMG,
        "loss_names": {**ALL_LOSS_NAMES, "vqa": 1},
        "vqav2_label_size": 4,
    })
    return cfg


@pytest.fixture(scope="session")
def two_tower_mrpc_config():
    cfg = _base_config()
    cfg.update({
        "model_type": "two-tower",
        "image_size": TWO_TOWER_IMG,
        "original_image_size": TWO_TOWER_IMG,
        "loss_names": {**ALL_LOSS_NAMES, "mrpc": 1},
    })
    return cfg


# ---------------------------------------------------------------------------
# Synthetic batch factories
# ---------------------------------------------------------------------------

def _make_pretrain_batch(img_size):
    """Standard pretrain batch: image + text + false image for ITM + MLM fields."""
    return {
        "image": [torch.randn(BS, 3, img_size, img_size)],
        "false_image_0": [torch.randn(BS, 3, img_size, img_size)],
        "text": ["hello world"] * BS,
        "text_ids": torch.randint(1, VOCAB, (BS, TEXT_LEN)),
        "text_labels": torch.full((BS, TEXT_LEN), -100, dtype=torch.long),
        "text_masks": torch.ones(BS, TEXT_LEN, dtype=torch.long),
        # MLM variants (used when mask_text=True)
        "text_ids_mlm": torch.randint(1, VOCAB, (BS, TEXT_LEN)),
        "text_labels_mlm": torch.cat([
            torch.randint(1, VOCAB, (BS, 3)),
            torch.full((BS, TEXT_LEN - 3), -100, dtype=torch.long),
        ], dim=1),
    }


@pytest.fixture
def one_tower_batch():
    return _make_pretrain_batch(ONE_TOWER_IMG)


@pytest.fixture
def two_tower_batch():
    return _make_pretrain_batch(TWO_TOWER_IMG)


@pytest.fixture
def snli_batch(two_tower_batch):
    return {
        **two_tower_batch,
        "labels": [0, 1],
        "table_name": ["snli_dev", "snli_test"],
    }


@pytest.fixture
def nlvr2_batch():
    """NLVR2 needs image_0 and image_1 keys for the two separate infer calls."""
    base = _make_pretrain_batch(TWO_TOWER_IMG)
    base["image_0"] = [torch.randn(BS, 3, TWO_TOWER_IMG, TWO_TOWER_IMG)]
    base["image_1"] = [torch.randn(BS, 3, TWO_TOWER_IMG, TWO_TOWER_IMG)]
    base["answers"] = [0, 1]
    base["table_name"] = ["nlvr2_dev", "nlvr2_test"]
    return base


@pytest.fixture
def ref_batch():
    """Ref batch: BS*MAX_BB crops stacked, one target label per example."""
    total = BS * MAX_BB
    return {
        "image": [torch.randn(total, 3, TWO_TOWER_IMG, TWO_TOWER_IMG)],
        "text_ids": torch.randint(1, VOCAB, (total, TEXT_LEN)),
        "text_labels": torch.full((total, TEXT_LEN), -100, dtype=torch.long),
        "text_masks": torch.ones(total, TEXT_LEN, dtype=torch.long),
        "target": torch.randint(0, MAX_BB, (BS,)),
    }


@pytest.fixture
def vqa_batch(two_tower_batch):
    """VQA batch: pretrain batch + per-example label/score lists matching the
    modern data layer's pass-through schema."""
    return {
        **two_tower_batch,
        "vqa_labels": [[0, 1], [2]],
        "vqa_scores": [[1.0, 0.3], [1.0]],
    }


@pytest.fixture
def mrpc_batch():
    """MRPC batch: text-only, infer_text_only unpacks it into the HF text
    encoder, so only `input_ids` / `attention_mask` are needed."""
    return {
        "input_ids": torch.randint(1, VOCAB, (BS, TEXT_LEN)),
        "attention_mask": torch.ones(BS, TEXT_LEN, dtype=torch.long),
        "label": torch.tensor([0, 1]),
    }
