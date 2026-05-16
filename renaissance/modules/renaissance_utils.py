import torch
# import random
import os

from torch.optim import AdamW
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
# from .dist_utils import all_gather
from .objectives import compute_irtr_recall
from ..gadgets.my_metrics import Accuracy, VQAScore, Scalar, IoU
# from torchmetrics import F1Score
from torchmetrics.classification import BinaryF1Score, MatthewsCorrCoef

# Creat a set_attribute type function to gneralize this
def set_metrics(pl_module):
    for split in ["train", "val"]:
        for k, v in pl_module.config["loss_names"].items():
            if v <= 0:
                continue
            if k == "vqa":
                setattr(pl_module, f"{split}_vqa_score", VQAScore())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            elif k == "ref":
                setattr(pl_module, f"{split}_ref_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            elif k == "ref2":
                setattr(pl_module, f"{split}_ref2_iou", IoU())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            elif k == "nlvr2":
                if split == "train":
                    setattr(pl_module, f"train_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"train_{k}_loss", Scalar())
                else:
                    setattr(pl_module, f"dev_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"dev_{k}_loss", Scalar())
                    setattr(pl_module, f"test_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"test_{k}_loss", Scalar())
            elif k == "snli":
                if split == "train":
                    setattr(pl_module, f"train_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"train_{k}_loss", Scalar())
                else:
                    setattr(pl_module, f"dev_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"dev_{k}_loss", Scalar())
                    setattr(pl_module, f"test_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"test_{k}_loss", Scalar())
            elif k == "irtr":
                setattr(pl_module, f"{split}_irtr_loss", Scalar())
            elif k == "mppd" or k == "mpfr":
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            elif k == "itm":
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            elif k == "mrpc":
                # f1 = BinaryF1Score()
                setattr(pl_module, f"{split}_mrpc_f1", BinaryF1Score())
                setattr(pl_module, f"{split}_mrpc_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            elif k == "rte":
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            elif k == "wnli":
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            elif k == "sst2":
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            elif k == "qqp":
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            elif k == "qnli":
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            elif k == "mnli":
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            elif k == "cola":
                setattr(pl_module, f"{split}_{k}_mcc", MatthewsCorrCoef(task='binary'))
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            elif k == "cifar10":
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            else:
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())


def epoch_wrapup(pl_module, phase: str, epoch: int = 0, log_dir: str = "") -> dict:
    """Compute and reset all epoch-level metrics.  Returns a flat dict of
    ``{metric_name: float}`` for the caller (Trainer) to log to TensorBoard."""
    metrics: dict = {}
    the_metric = 0

    if pl_module.config["get_recall_metric"] and phase == "val":
        (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10) = compute_irtr_recall(pl_module)
        print((ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10))
        for key, val in [
            ("recalls/ir_r1", ir_r1), ("recalls/ir_r5", ir_r5),
            ("recalls/ir_r10", ir_r10), ("recalls/tr_r1", tr_r1),
            ("recalls/tr_r5", tr_r5), ("recalls/tr_r10", tr_r10),
        ]:
            metrics[key] = val.item() if torch.is_tensor(val) else val
        the_metric += ir_r1.item() + tr_r1.item()

    for loss_name, v in pl_module.config["loss_names"].items():
        if v <= 0:
            continue

        value = 0
        
        # Create function to minimeze these steps
        if loss_name == "vqa":
            value = getattr(pl_module, f"{phase}_{loss_name}_score").compute()
            metrics[f"{loss_name}/{phase}/score_epoch"] = _to_float(value)
            getattr(pl_module, f"{phase}_{loss_name}_score").reset()
            loss_val = getattr(pl_module, f"{phase}_{loss_name}_loss").compute()
            metrics[f"{loss_name}/{phase}/loss_epoch"] = _to_float(loss_val)
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
        elif loss_name == 'ref':
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            metrics[f"{loss_name}/{phase}/accuracy_epoch"] = _to_float(value)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
            loss_val = getattr(pl_module, f"{phase}_{loss_name}_loss").compute()
            metrics[f"{loss_name}/{phase}/loss_epoch"] = _to_float(loss_val)
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
            if log_dir:
                _write_eval(log_dir, epoch, phase, loss_val, accuracy=value)
        elif loss_name == 'ref2':
            value = getattr(pl_module, f"{phase}_{loss_name}_iou").compute()
            metrics[f"{loss_name}/{phase}/iou_epoch"] = _to_float(value)
            getattr(pl_module, f"{phase}_{loss_name}_iou").reset()
            loss_val = getattr(pl_module, f"{phase}_{loss_name}_loss").compute()
            metrics[f"{loss_name}/{phase}/loss_epoch"] = _to_float(loss_val)
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
            if log_dir:
                _write_eval(log_dir, epoch, phase, loss_val, iou=value)
        elif loss_name in ("nlvr2", "snli"):
            if phase == "train":
                value = getattr(pl_module, f"train_{loss_name}_accuracy").compute()
                metrics[f"{loss_name}/train/accuracy_epoch"] = _to_float(value)
                getattr(pl_module, f"train_{loss_name}_accuracy").reset()
                loss_val = getattr(pl_module, f"train_{loss_name}_loss").compute()
                metrics[f"{loss_name}/train/loss_epoch"] = _to_float(loss_val)
                getattr(pl_module, f"train_{loss_name}_loss").reset()
            else:
                for split in ("test", "dev"):
                    v = getattr(pl_module, f"{split}_{loss_name}_accuracy").compute()
                    metrics[f"{loss_name}/{split}/accuracy_epoch"] = _to_float(v)
                    getattr(pl_module, f"{split}_{loss_name}_accuracy").reset()
                    lv = getattr(pl_module, f"{split}_{loss_name}_loss").compute()
                    metrics[f"{loss_name}/{split}/loss_epoch"] = _to_float(lv)
                    getattr(pl_module, f"{split}_{loss_name}_loss").reset()
                value = metrics.get(f"{loss_name}/test/accuracy_epoch", 0)
        elif loss_name == 'mrpc':
            value = getattr(pl_module, f"{phase}_{loss_name}_f1").compute()
            metrics[f"{loss_name}/{phase}/f1_epoch"] = _to_float(value)
            getattr(pl_module, f"{phase}_{loss_name}_f1").reset()
            acc = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            metrics[f"{loss_name}/{phase}/accuracy_epoch"] = _to_float(acc)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
            loss_val = getattr(pl_module, f"{phase}_{loss_name}_loss").compute()
            metrics[f"{loss_name}/{phase}/loss_epoch"] = _to_float(loss_val)
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
            if log_dir:
                _write_eval(log_dir, epoch, phase, loss_val, f1=value, accuracy=acc)
        elif loss_name == 'cola':
            value = getattr(pl_module, f"{phase}_{loss_name}_mcc").compute()
            metrics[f"{loss_name}/{phase}/mcc_epoch"] = _to_float(value)
            getattr(pl_module, f"{phase}_{loss_name}_mcc").reset()
            loss_val = getattr(pl_module, f"{phase}_{loss_name}_loss").compute()
            metrics[f"{loss_name}/{phase}/loss_epoch"] = _to_float(loss_val)
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
            if log_dir:
                _write_eval(log_dir, epoch, phase, loss_val, mcc=value)
        elif loss_name == "irtr":
            loss_val = getattr(pl_module, f"{phase}_irtr_loss").compute()
            metrics[f"{loss_name}/{phase}/irtr_loss_epoch"] = _to_float(loss_val)
            getattr(pl_module, f"{phase}_irtr_loss").reset()
        elif loss_name in ("mppd", "mpfr"):
            loss_val = getattr(pl_module, f"{phase}_{loss_name}_loss").compute()
            metrics[f"{loss_name}/{phase}/loss_epoch"] = _to_float(loss_val)
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
        else:
            # itm, mlm, rte, wnli, sst2, qqp, qnli, mnli, cifar10, and fallback
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            metrics[f"{loss_name}/{phase}/accuracy_epoch"] = _to_float(value)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
            loss_val = getattr(pl_module, f"{phase}_{loss_name}_loss").compute()
            metrics[f"{loss_name}/{phase}/loss_epoch"] = _to_float(loss_val)
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
            if log_dir and loss_name not in ("itm", "mlm"):
                _write_eval(log_dir, epoch, phase, loss_val, accuracy=value)

        the_metric += _to_float(value)

    metrics[f"{phase}/the_metric"] = the_metric
    return metrics


def _to_float(v) -> float:
    return v.item() if torch.is_tensor(v) else float(v)


def _write_eval(log_dir: str, epoch: int, phase: str, loss, **extra):
    file_path = os.path.join(log_dir, "eval.txt")
    with open(file_path, "a") as f:
        f.write(f"Epoch: {epoch}, Loss on {phase}: {loss}\n")
        for k, v in extra.items():
            f.write(f"Epoch: {epoch}, {k} on {phase}: {v}\n")
        f.write("\n")


def set_task(pl_module):
    pl_module.current_tasks = [
        k for k, v in pl_module.config["loss_names"].items() if v > 0
    ]


def set_schedule(model, config, max_steps: int):
    """Build optimizer + LR scheduler.  Returns (optimizer, scheduler)."""
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
    head_names = [
        # RenaissanceModel keeps all task heads in an nn.ModuleDict named
        # `heads`, so `heads.` is the modern head-param prefix.
        "heads.",
        # Legacy RenaissanceTransformer head attribute names (kept so the
        # legacy model still gets the head LR multiplier until Phase 7).
        "vqa_classifier", "nlvr2_classifier", "mlm_score", "itm_score", "snli_classifier",
        "ref_classifier", "ref2_classifier", "mrpc_classifier", "rte_classifier",
        "wnli_classifier", "sst2_classifier", "qqp_classifier", "qnli_classifier",
        "mnli_classifier", "cola_classifier", "cifar10_classifier",
        "image_classification_pooler", "text_classification_pooler",
    ]
    cross_modal_names = ['cross_modal']
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
                if any(nd in n for nd in no_decay) and any(bb in n for bb in head_names)
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
