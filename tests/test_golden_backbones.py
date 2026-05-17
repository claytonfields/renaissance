"""
Phase 8 — golden / determinism safety net for backbone internals.

Two guards for the encoder-internals modernization:

1. Determinism: same seed + same input ⇒ bitwise-identical forward across
   two independent builds. Environment-independent; catches a refactor
   that perturbs the construction/computation order.

2. Golden snapshot: the current forward output is frozen to
   `tests/golden/backbone_<type>.pt`. A behavior-bearing refactor (e.g.
   re-routing HF loading) must reproduce it. Because HF init/forward can
   differ across `transformers` versions, the snapshot records the lib
   versions and the comparison is *skipped* when they differ — so it
   enforces equivalence within the environment a developer refactors in
   (local), while CI (often a different transformers) relies on the broad
   behavioral suite + the determinism guard above. Regenerate by deleting
   the snapshot file and rerunning.

Tiny pinned configs; CPU-only.
"""

import pathlib

import pytest
import torch
import transformers

from renaissance.modeling.backbones import build_backbone

GOLDEN_DIR = pathlib.Path(__file__).parent / "golden"
SEED = 1234
BS = 2
TEXT_LEN = 8


def _two_tower_cfg():
    return {
        "model_type": "two-tower",
        "image_encoder": "facebook/deit-tiny-patch16-224",
        "text_encoder": "google/electra-small-discriminator",
        "random_init_vision_encoder": True,
        "random_init_text_encoder": True,
        "image_encoder_manual_configuration": False,
        "text_encoder_manual_configuration": False,
        "freeze_image_encoder": False,
        "freeze_text_encoder": False,
        "freeze_cross_modal_layers": False,
        "cross_layer_hidden_size": 32,
        "num_cross_layers": 2,
        "num_cross_layer_heads": 4,
        "cross_layer_mlp_ratio": 4,
        "cross_layer_drop_rate": 0.0,
        "image_size": 224,
        "max_text_len": TEXT_LEN,
        "vocab_size": 30522,
    }


def _one_tower_cfg():
    return {
        "model_type": "one-tower",
        "encoder": "google/electra-small-discriminator",
        "pooler_type": "double",
        "random_init_encoder": True,
        "encoder_manual_configuration": False,
        "image_size": 32,
        "patch_size": 16,
        "max_text_len": TEXT_LEN,
        "vocab_size": 30522,
        "drop_rate": 0.0,
    }


def _batch(img_size):
    g = torch.Generator().manual_seed(99)
    return {
        "image": [torch.randn(BS, 3, img_size, img_size, generator=g)],
        "text_ids": torch.randint(1, 30522, (BS, TEXT_LEN), generator=g),
        "text_labels": torch.full((BS, TEXT_LEN), -100, dtype=torch.long),
        "text_masks": torch.ones(BS, TEXT_LEN, dtype=torch.long),
        "text_ids_mlm": torch.randint(1, 30522, (BS, TEXT_LEN), generator=g),
        "text_labels_mlm": torch.full((BS, TEXT_LEN), -100, dtype=torch.long),
    }


def _forward(cfg, img_size):
    torch.manual_seed(SEED)
    bb = build_backbone(cfg)
    bb.eval()
    with torch.no_grad():
        out = bb(_batch(img_size))
    return {
        "pooled": out.pooled,
        "text_tokens": out.text_tokens,
        "image_tokens": out.image_tokens,
    }


CASES = {
    "two_tower": (_two_tower_cfg, 224),
    "one_tower": (_one_tower_cfg, 32),
}


@pytest.mark.parametrize("name", list(CASES))
def test_determinism(name):
    cfg_fn, img = CASES[name]
    a = _forward(cfg_fn(), img)
    b = _forward(cfg_fn(), img)
    for k in a:
        assert torch.equal(a[k], b[k]), f"{name}: {k} not deterministic under fixed seed"


@pytest.mark.parametrize("name", list(CASES))
def test_golden_snapshot(name):
    cfg_fn, img = CASES[name]
    GOLDEN_DIR.mkdir(exist_ok=True)
    path = GOLDEN_DIR / f"backbone_{name}.pt"
    current = _forward(cfg_fn(), img)

    if not path.exists():
        torch.save(
            {
                "transformers_version": transformers.__version__,
                "torch_version": torch.__version__,
                **current,
            },
            path,
        )
        pytest.skip(f"generated golden snapshot {path.name}; rerun to compare")

    # Locally-generated, trusted file with version-string metadata.
    ref = torch.load(path, weights_only=False)
    if ref.get("transformers_version") != transformers.__version__ or ref.get(
        "torch_version"
    ) != torch.__version__:
        pytest.skip(
            f"{name}: golden generated under transformers="
            f"{ref.get('transformers_version')} / torch={ref.get('torch_version')}, "
            f"running {transformers.__version__} / {torch.__version__}"
        )

    for k in current:
        assert torch.allclose(current[k], ref[k], atol=1e-5), (
            f"{name}: {k} diverged from golden snapshot"
        )
