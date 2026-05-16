"""
Phase 2 — unit tests for `renaissance.modeling.heads`.

Shape/contract tests for the cleaned-up heads. The cross-rewrite
behavioral net is `test_contract.py`; this pins the new heads in
isolation, including the equivalences that justify collapsing the three
legacy `*ClassificationHead` classes into one `LinearClsHead`.
"""

import torch
import torch.nn as nn

from renaissance.modeling.heads import (
    ItmHead,
    LinearClsHead,
    MlmHead,
    Pooler,
    init_weights,
)

B, T, H, V = 4, 7, 16, 100


def _finite(t):
    return t.isfinite().all().item()


class TestPooler:

    def test_takes_cls_and_projects(self):
        pooler = Pooler(H)
        out = pooler(torch.randn(B, T, H))
        assert out.shape == (B, H)
        assert _finite(out)

    def test_uses_only_token_zero(self):
        pooler = Pooler(H)
        x = torch.randn(B, T, H)
        x2 = x.clone()
        x2[:, 1:] = 999.0  # perturb every non-CLS token
        assert torch.equal(pooler(x), pooler(x2))


class TestItmHead:

    def test_shape(self):
        head = ItmHead(H)
        assert head(torch.randn(B, H)).shape == (B, 2)


class TestMlmHead:

    def test_shape(self):
        head = MlmHead(H, V)
        out = head(torch.randn(B, T, H))
        assert out.shape == (B, T, V)
        assert _finite(out)

    def test_weight_tying(self):
        emb = nn.Embedding(V, H)
        head = MlmHead(H, V, weight=emb.weight)
        assert head.decoder.weight is emb.weight


class TestLinearClsHead:

    def test_multimodal_no_pool(self):
        # VQA/SNLI/ref/ref2: input is already-pooled cls_feats.
        head = LinearClsHead(H, num_labels=3, pool=False)
        assert head(torch.randn(B, H)).shape == (B, 3)

    def test_unimodal_pool(self):
        # GLUE/CIFAR-10: input is a token sequence, pool the CLS token.
        head = LinearClsHead(H, num_labels=2, pool=True)
        assert head(torch.randn(B, T, H)).shape == (B, 2)

    def test_nlvr2_concat_widths(self):
        # NLVR2 feeds concat(2 pooled features); intermediate stays pooled_dim.
        head = LinearClsHead(2 * H, num_labels=2, hidden_dim=H, pool=False)
        assert head.dense.in_features == 2 * H
        assert head.dense.out_features == H
        assert head(torch.randn(B, 2 * H)).shape == (B, 2)

    def test_bbox_regression_width(self):
        # ref2: 4-output regression head.
        head = LinearClsHead(H, num_labels=4, pool=False)
        assert head(torch.randn(B, H)).shape == (B, 4)

    def test_pool_slices_token_zero(self):
        head = LinearClsHead(H, num_labels=2, pool=True)
        x = torch.randn(B, T, H)
        x2 = x.clone()
        x2[:, 1:] = -42.0
        assert torch.equal(head(x), head(x2))


class TestInitWeights:

    def test_linear_bias_zeroed_weight_scaled(self):
        lin = nn.Linear(64, 64)
        lin.apply(init_weights)
        assert torch.equal(lin.bias.data, torch.zeros_like(lin.bias.data))
        assert abs(lin.weight.data.std().item() - 0.02) < 0.01

    def test_layernorm_reset(self):
        ln = nn.LayerNorm(64)
        ln.weight.data.fill_(0.5)
        ln.bias.data.fill_(0.5)
        ln.apply(init_weights)
        assert torch.equal(ln.weight.data, torch.ones_like(ln.weight.data))
        assert torch.equal(ln.bias.data, torch.zeros_like(ln.bias.data))
