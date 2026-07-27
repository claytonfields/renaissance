#!/usr/bin/env python
"""Generate the v2 golden shape fixture from the live 1.3 two-tower backbone.

Run from the repo root in a 1.3 environment:

    python scripts/gen_v2_golden_fixture.py

Builds the issue-#2 target composition (dino-vits16 + electra-small +
cross-attention, random-init so no weight downloads), runs a seeded forward,
asserts the recorded shapes against the live outputs, and writes
v2/tests/golden/two_tower_shapes.json. The JSON is generated evidence —
never hand-edit it; re-run this script instead.
"""

import json
import pathlib
import subprocess
import sys

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # import 1.3 from the checkout, installed or not

from renaissance.modeling.backbones import build_backbone  # noqa: E402

SEED = 1234
BATCH_SIZE = 2
TEXT_LEN = 16
IMAGE_SIZE = 224
VOCAB_SIZE = 30522

CONFIG_13 = {
    "model_type": "two-tower",
    "image_encoder": "facebook/dino-vits16",
    "text_encoder": "google/electra-small-discriminator",
    "random_init_vision_encoder": True,
    "random_init_text_encoder": True,
    "image_encoder_manual_configuration": False,
    "text_encoder_manual_configuration": False,
    "freeze_image_encoder": False,
    "freeze_text_encoder": False,
    "freeze_cross_modal_layers": False,
    "cross_layer_hidden_size": 384,
    "num_cross_layers": 2,
    "num_cross_layer_heads": 6,
    "cross_layer_mlp_ratio": 4,
    "cross_layer_drop_rate": 0.0,
    "image_size": IMAGE_SIZE,
    "max_text_len": TEXT_LEN,
    "vocab_size": VOCAB_SIZE,
}

# The same composition expressed as a v2 ModelConfig dict; the v2 golden test
# builds its model straight from this block so the two can never drift.
MODEL_V2 = {
    "encoders": {
        "text": {"name": "hf-text", "args": {"hub_id": CONFIG_13["text_encoder"], "random_init": True}},
        "image": {"name": "hf-image", "args": {"hub_id": CONFIG_13["image_encoder"], "random_init": True}},
    },
    "fusion": {
        "name": "cross-attention",
        "args": {
            "hidden_size": CONFIG_13["cross_layer_hidden_size"],
            "num_layers": CONFIG_13["num_cross_layers"],
            "num_heads": CONFIG_13["num_cross_layer_heads"],
            "mlp_ratio": CONFIG_13["cross_layer_mlp_ratio"],
            "drop_rate": CONFIG_13["cross_layer_drop_rate"],
        },
    },
    "tasks": [],
}


def main() -> None:
    torch.manual_seed(SEED)
    backbone = build_backbone(CONFIG_13)
    backbone.eval()

    batch = {
        "image": [torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)],
        "text_ids": torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, TEXT_LEN)),
        "text_labels": torch.full((BATCH_SIZE, TEXT_LEN), -100, dtype=torch.long),
        "text_masks": torch.ones(BATCH_SIZE, TEXT_LEN, dtype=torch.long),
    }
    with torch.no_grad():
        out = backbone(batch)

    expected_shapes = {
        "pooled": list(out.pooled.shape),
        "text_tokens": list(out.text_tokens.shape),
        "image_tokens": list(out.image_tokens.shape),
    }
    expected_dims = {"pooled_dim": backbone.pooled_dim, "token_dim": backbone.token_dim}

    h = CONFIG_13["cross_layer_hidden_size"]
    n_patches = (IMAGE_SIZE // 16) ** 2 + 1  # dino-vits16: 196 patches + CLS
    assert expected_shapes["pooled"] == [BATCH_SIZE, 2 * h], expected_shapes
    assert expected_shapes["text_tokens"] == [BATCH_SIZE, TEXT_LEN, h], expected_shapes
    assert expected_shapes["image_tokens"] == [BATCH_SIZE, n_patches, h], expected_shapes
    assert expected_dims == {"pooled_dim": 2 * h, "token_dim": h}, expected_dims

    import transformers

    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    fixture = {
        "generated_by": (
            f"scripts/gen_v2_golden_fixture.py @ {sha or 'unknown'}, "
            f"transformers {transformers.__version__}, torch {torch.__version__}"
        ),
        "model": MODEL_V2,
        "batch": {
            "batch_size": BATCH_SIZE,
            "text_len": TEXT_LEN,
            "image_size": IMAGE_SIZE,
            "n_images": 1,
            "vocab_size": VOCAB_SIZE,
        },
        "expected_shapes": expected_shapes,
        "expected_dims": expected_dims,
    }

    out_path = pathlib.Path(__file__).resolve().parent.parent / "v2" / "tests" / "golden" / "two_tower_shapes.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"wrote {out_path}")
    for name, shape in expected_shapes.items():
        print(f"  {name}: {shape}")


if __name__ == "__main__":
    main()
