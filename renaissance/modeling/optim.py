"""
Optimizer + LR-schedule construction.

Relocated from the deleted `renaissance/modules/renaissance_utils.py`
(the rest of which — `set_task`/`set_metrics`/`epoch_wrapup` — was dead
after the modeling rewrite; `RenaissanceModel` owns those now).

Parameter groups: a `lr_mult_head` multiplier for task-head params (the
`heads.` ModuleDict prefix) and `lr_mult_cross_modal` for cross-modal
fusion params, each split into weight-decay / no-decay subgroups.
"""

import torch
from torch.optim import AdamW
from transformers import (
    get_cosine_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)


def set_schedule(model, config, max_steps: int):
    """Build optimizer + LR scheduler. Returns (optimizer, scheduler)."""
    lr = config["learning_rate"]
    wd = config["weight_decay"]

    no_decay = [
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "norm.bias",
        "norm.weight",
        "norm1.bias",
        "norm1.weight",
        "norm2.bias",
        "norm2.weight",
    ]
    # RenaissanceModel keeps every task head in an nn.ModuleDict named
    # `heads`, so `heads.` is the head-param prefix.
    head_names = ["heads."]
    cross_modal_names = ["cross_modal"]
    lr_mult_head = config["lr_mult_head"]
    lr_mult_cross_modal = config["lr_mult_cross_modal"]
    end_lr = config["end_lr"]
    decay_power = config["decay_power"]
    optim_type = config["optim_type"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
                and not any(ht in n for ht in cross_modal_names)
            ],
            "weight_decay": wd,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
                and not any(ht in n for ht in cross_modal_names)
            ],
            "weight_decay": 0.0,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
                and any(bb in n for bb in head_names)
                and not any(ht in n for ht in cross_modal_names)
            ],
            "weight_decay": wd,
            "lr": lr * lr_mult_head,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
                and any(bb in n for bb in head_names)
                and not any(ht in n for ht in cross_modal_names)
            ],
            "weight_decay": 0.0,
            "lr": lr * lr_mult_head,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
                and any(ht in n for ht in cross_modal_names)
            ],
            "weight_decay": wd,
            "lr": lr * lr_mult_cross_modal,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
                and any(ht in n for ht in cross_modal_names)
            ],
            "weight_decay": 0.0,
            "lr": lr * lr_mult_cross_modal,
        },
    ]

    if optim_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98)
        )
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
    elif optim_type == "sgd":
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)

    warmup_steps = config["warmup_steps"]
    if isinstance(warmup_steps, float):
        warmup_steps = int(max_steps * warmup_steps)

    if decay_power == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps,
        )
    else:
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            lr_end=end_lr,
            power=decay_power,
        )

    return optimizer, scheduler
