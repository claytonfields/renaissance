"""
Clean classification / decoder heads for the modeling layer (Phase 2).

Replaces `renaissance/modules/heads.py`, which carried two near-identical
`*ClassificationHead` classes (differing only in whether they sliced the
CLS token), three unused `RefRes*` variants, an inlined `BertLayer`, an
inlined `BertPredictionHeadTransform`, and commented-out dead code.

All of VQA / SNLI / ref / ref2 / NLVR2 / GLUE / CIFAR-10 used the same
Linear→LayerNorm→GELU→Linear shape — they only ever differed in input
width, intermediate width, and whether to pool the CLS token. That is now
one `LinearClsHead`. `MlmHead` and `ItmHead` stay dedicated.

Additive: the legacy `renaissance/modules/heads.py` is untouched until
the Phase 7 cutover.
"""

import torch
import torch.nn as nn


def init_weights(module: nn.Module) -> None:
    """Truncated-normal-ish init matching the legacy `objectives.init_weights`
    so head initialisation behaviour is preserved across the rewrite."""
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class Pooler(nn.Module):
    """Take the CLS token, project, tanh. Matches the legacy `Pooler`."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        return self.activation(self.dense(hidden_states[:, 0]))


class ItmHead(nn.Module):
    """Image-text matching: binary classifier on the pooled feature."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 2)

    def forward(self, x):
        return self.fc(x)


class MlmHead(nn.Module):
    """ELECTRA/BERT-style masked-LM head: transform → decoder + bias.

    `weight` optionally ties the decoder to an embedding matrix. Replaces
    the legacy `MLMHead`, which took a `config` dict and built a throwaway
    `BertConfig` just to reach `BertPredictionHeadTransform`.
    """

    def __init__(self, hidden_size: int, vocab_size: int, layer_norm_eps: float = 1e-12, weight=None):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        if weight is not None:
            self.decoder.weight = weight

    def forward(self, x):
        x = self.layer_norm(self.activation(self.dense(x)))
        return self.decoder(x) + self.bias


class LinearClsHead(nn.Module):
    """Generic classification / regression head.

    Shape: ``Linear(in_dim → hidden_dim) → LayerNorm → GELU →
    Linear(hidden_dim → num_labels)``.

    Parameters
    ----------
    in_dim
        Input feature width.
    num_labels
        Output width — class count, or 4 for bbox regression (ref2).
    hidden_dim
        Intermediate width; defaults to ``in_dim``. NLVR2 feeds the
        concatenation of two pooled features (``in_dim = 2 * pooled_dim``)
        but keeps the intermediate at ``pooled_dim``.
    pool
        If True, slice ``features[:, 0]`` (the CLS token of a sequence)
        before projecting. Multimodal heads get an already-pooled feature
        (``pool=False``); text-only / image-only heads get a token
        sequence (``pool=True``).
    """

    def __init__(self, in_dim: int, num_labels: int, hidden_dim: int = None, pool: bool = False):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        self.pool = pool
        self.dense = nn.Linear(in_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.GELU()
        self.out_proj = nn.Linear(hidden_dim, num_labels)

    def forward(self, features):
        x = features[:, 0] if self.pool else features
        x = self.dense(x)
        x = self.layer_norm(x)
        x = self.activation(x)
        return self.out_proj(x)
