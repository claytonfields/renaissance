"""Shared fixtures: seeded synthetic batches and the golden-fixture loader."""

import json
import pathlib

import torch

GOLDEN_DIR = pathlib.Path(__file__).parent / "golden"
SEED = 1234


def load_golden_fixture(name: str = "two_tower_shapes.json") -> dict:
    return json.loads((GOLDEN_DIR / name).read_text())


def make_batch(
    *,
    batch_size: int = 2,
    text_len: int = 16,
    image_size: int = 224,
    n_images: int = 1,
    vocab_size: int = 30522,
    seed: int = SEED,
) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {
        "images": torch.randn(batch_size, n_images, 3, image_size, image_size, generator=g),
        "text_ids": torch.randint(0, vocab_size, (batch_size, text_len), generator=g),
        "text_masks": torch.ones(batch_size, text_len, dtype=torch.long),
    }


# Tiny random-init specs for tests that don't need the golden architecture.
# AutoConfig JSONs are fetched from the Hub (cached); no weight downloads.
TINY_TEXT_ARGS = {
    "hub_id": "google/electra-small-discriminator",
    "random_init": True,
    "overrides": {
        "num_hidden_layers": 1,
        "hidden_size": 64,
        "num_attention_heads": 4,
        "intermediate_size": 128,
    },
}
TINY_IMAGE_ARGS = {
    "hub_id": "facebook/dino-vits16",
    "random_init": True,
    "overrides": {
        "num_hidden_layers": 1,
        "hidden_size": 48,
        "num_attention_heads": 4,
        "intermediate_size": 96,
        "image_size": 32,
    },
}
TINY_FUSION_ARGS = {"hidden_size": 32, "num_layers": 1, "num_heads": 4, "mlp_ratio": 2, "drop_rate": 0.0}
