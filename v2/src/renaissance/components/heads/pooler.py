"""Pooler and weight init, ported verbatim from the 1.3 line (`modeling/heads.py`)."""

from torch import nn


def init_weights(module: nn.Module) -> None:
    """Truncated-normal-ish init matching the legacy ``objectives.init_weights``
    so head initialisation behaviour is preserved across the rewrite."""
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class Pooler(nn.Module):
    """Take the CLS token, project, tanh."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        return self.activation(self.dense(hidden_states[:, 0]))
