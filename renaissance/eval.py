"""
Standalone evaluation harness for RenaissanceTransformer.

Usage
-----
Programmatic:
    from renaissance.eval import evaluate
    metrics = evaluate(model, dataloader, task="snli")

CLI:
    python -m renaissance.eval \\
        --checkpoint /path/to/ckpt \\
        --task snli --split val \\
        --data_root data/arrow/ \\
        --output results.json
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Optional

import torch

from .modules import renaissance_utils
from .modules import objectives


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: _to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(_to_device(v, device) for v in batch)
    return batch


# ---------------------------------------------------------------------------
# Evaluator base + per-task subclasses
# ---------------------------------------------------------------------------

class Evaluator:
    """Run inference over a dataloader and aggregate task metrics.

    Subclasses register themselves automatically via __init_subclass__.
    Use the class attribute ``task`` to identify the task name.
    """

    _REGISTRY: Dict[str, type] = {}

    task: str = ""

    def __init__(self, model):
        self.model = model

    def __init_subclass__(cls, task: str = "", **kwargs):
        super().__init_subclass__(**kwargs)
        if task:
            cls.task = task
            Evaluator._REGISTRY[task] = cls

    # ---- override in subclasses ----

    def process_batch(self, batch) -> None:
        raise NotImplementedError

    # ---- shared aggregation ----

    def aggregate(self, phase: str = "val") -> Dict[str, Any]:
        """Compute & reset epoch-level metrics via the shared epoch_wrapup."""
        return renaissance_utils.epoch_wrapup(self.model, phase=phase)

    # ---- callable interface ----

    def __call__(
        self,
        dataloader,
        device: Optional[torch.device] = None,
        phase: str = "val",
    ) -> Dict[str, Any]:
        if device is None:
            device = self.model.device
        self.model.eval()
        self.model.current_tasks = [self.task]
        with torch.no_grad():
            for batch in dataloader:
                batch = _to_device(batch, device)
                self.model._log_buffer = {}
                self.process_batch(batch)
        return self.aggregate(phase=phase)


class MLMEvaluator(Evaluator, task="mlm"):
    def process_batch(self, batch):
        objectives.compute_mlm(self.model, batch)


class ITMEvaluator(Evaluator, task="itm"):
    def process_batch(self, batch):
        objectives.compute_itm(self.model, batch)


class VQAEvaluator(Evaluator, task="vqa"):
    def process_batch(self, batch):
        objectives.compute_vqa(self.model, batch)


class NLVR2Evaluator(Evaluator, task="nlvr2"):
    def process_batch(self, batch):
        objectives.compute_nlvr2(self.model, batch)


class SNLIEvaluator(Evaluator, task="snli"):
    def process_batch(self, batch):
        objectives.compute_snli(self.model, batch)


class RefEvaluator(Evaluator, task="ref"):
    def process_batch(self, batch):
        objectives.compute_ref(self.model, batch)


class Ref2Evaluator(Evaluator, task="ref2"):
    def process_batch(self, batch):
        objectives.compute_ref2(self.model, batch)


class IRTREvaluator(Evaluator, task="irtr"):
    """Evaluates the IRTR ranking loss per batch.

    Note: for R@1/R@5/R@10 recall metrics over the full corpus use
    ``objectives.compute_irtr_recall`` with dedicated text + image dataloaders.
    """
    def process_batch(self, batch):
        objectives.compute_irtr(self.model, batch)


class MRPCEvaluator(Evaluator, task="mrpc"):
    def process_batch(self, batch):
        objectives.compute_mrpc(self.model, batch)


# ---------------------------------------------------------------------------
# Top-level convenience function
# ---------------------------------------------------------------------------

def evaluate(
    model,
    dataloader,
    task: str,
    device: Optional[torch.device] = None,
    phase: str = "val",
) -> Dict[str, Any]:
    """Evaluate *model* on *dataloader* for *task* and return a metrics dict.

    Parameters
    ----------
    model:
        A ``RenaissanceTransformer`` instance with the correct task head
        instantiated (i.e. the corresponding ``loss_names[task] > 0``).
    dataloader:
        PyTorch DataLoader yielding batches in the format expected by the
        task's ``compute_*`` function.
    task:
        One of the registered task names (see ``Evaluator._REGISTRY``).
    device:
        Target device. Defaults to the device of the first model parameter.
    phase:
        ``"val"`` or ``"test"`` — controls which metric accumulators are
        read from the model (some tasks track dev/test splits separately).

    Returns
    -------
    dict
        Flat ``{metric_name: float}`` dict produced by ``epoch_wrapup``.
    """
    cls = Evaluator._REGISTRY.get(task)
    if cls is None:
        raise ValueError(
            f"No evaluator for task '{task}'. "
            f"Known tasks: {sorted(Evaluator._REGISTRY)}"
        )
    return cls(model)(dataloader, device=device, phase=phase)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    """python -m renaissance.eval --checkpoint <path> --task <task> ..."""
    parser = argparse.ArgumentParser(
        description="Evaluate a RenaissanceTransformer checkpoint on a downstream task."
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Local directory or HuggingFace Hub repo id (e.g. myuser/renaissance-pretrain).",
    )
    parser.add_argument(
        "--task", required=True, choices=sorted(Evaluator._REGISTRY),
        help="Downstream task to evaluate.",
    )
    parser.add_argument(
        "--split", default="val", choices=["val", "test"],
        help="Dataset split to evaluate on.",
    )
    parser.add_argument(
        "--data_root", default="data/arrow/",
        help="Root directory containing pre-converted Arrow files.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--output", default=None,
        help="Optional JSON file path to write results.",
    )
    args = parser.parse_args(argv)

    from .modules.renaissance_module import RenaissanceTransformer

    print(f"Loading model from {args.checkpoint!r} ...")
    model = RenaissanceTransformer.from_pretrained(args.checkpoint)
    model.eval()

    print(f"Building dataloader for task={args.task!r}, split={args.split!r} ...")
    dataloader = _build_dataloader(
        model, args.task, args.split, args.data_root,
        args.batch_size, args.num_workers,
    )

    print("Running evaluation ...")
    metrics = evaluate(model, dataloader, task=args.task, phase=args.split)

    print("\n=== Results ===")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    if args.output:
        with open(args.output, "w") as fh:
            json.dump(metrics, fh, indent=2)
        print(f"\nResults written to {args.output}")

    return metrics


def _build_dataloader(model, task, split, data_root, batch_size, num_workers):
    """Construct the appropriate dataloader for *task* and *split*.

    This is a thin dispatch layer; each dataset module is responsible for its
    own collate logic.  See DATA.md for dataset preparation instructions.
    """
    raise NotImplementedError(
        "Automatic dataloader construction is not yet implemented. "
        "Build the dataloader manually and call evaluate(model, dl, task) directly. "
        "See DATA.md for dataset preparation instructions."
    )


if __name__ == "__main__":
    main()
