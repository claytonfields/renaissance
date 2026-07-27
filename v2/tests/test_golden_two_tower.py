"""The Phase 1 done-criterion: compose a two-tower model
(dino-vits16 + electra-small + cross-attention) and match 1.3's forward
output shapes, recorded in golden/two_tower_shapes.json by
scripts/gen_v2_golden_fixture.py.

Shape parity is the spec — value parity is not (different init draw order
and implementation by design).
"""

import torch

from renaissance import ModelConfig, RenaissanceModel
from tests.conftest import SEED, load_golden_fixture, make_batch


def _build_and_forward(fixture: dict, seed: int = SEED):
    torch.manual_seed(seed)
    model = RenaissanceModel(ModelConfig.from_dict(fixture["model"]))
    model.eval()
    batch = make_batch(**fixture["batch"], seed=seed)
    with torch.no_grad():
        out = model(batch)
    return model, out


def test_golden_shapes_match_13():
    fixture = load_golden_fixture()
    model, out = _build_and_forward(fixture)

    expected = fixture["expected_shapes"]
    assert list(out.pooled.shape) == expected["pooled"]
    assert list(out.text_tokens.shape) == expected["text_tokens"]
    assert list(out.image_tokens.shape) == expected["image_tokens"]

    dims = fixture["expected_dims"]
    assert model.pooled_dim == dims["pooled_dim"]
    assert model.token_dim == dims["token_dim"]


def test_determinism():
    fixture = load_golden_fixture()
    _, a = _build_and_forward(fixture)
    _, b = _build_and_forward(fixture)
    assert torch.equal(a.pooled, b.pooled)
    assert torch.equal(a.text_tokens, b.text_tokens)
    assert torch.equal(a.image_tokens, b.image_tokens)
