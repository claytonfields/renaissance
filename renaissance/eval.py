"""
Standalone evaluation harness for RenaissanceModel.

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

Post-rewrite this is a thin wrapper: the model owns the metrics
(`RenaissanceModel.forward` updates them, `epoch_metrics` aggregates), so
the per-task `Evaluator` subclasses + `objectives.compute_*` dependency
the legacy harness needed are gone. `Evaluator._REGISTRY` and the
`evaluate()` signature are preserved for callers/tests.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Optional

import torch

from .modeling.tasks import TASK_REGISTRY


def _to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: _to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(_to_device(v, device) for v in batch)
    return batch


class Evaluator:
    """Kept for API/back-compat. The registry is the task registry; there
    are no per-task subclasses anymore — `evaluate()` drives the model
    directly."""

    _REGISTRY: Dict[str, Any] = TASK_REGISTRY


def evaluate(
    model,
    dataloader,
    task: str,
    device: Optional[torch.device] = None,
    phase: str = "val",
) -> Dict[str, Any]:
    """Evaluate *model* on *dataloader* for *task*; return a metrics dict.

    Runs the task through `RenaissanceModel.forward` (which updates the
    model-owned metrics) over the loader, then `model.epoch_metrics(phase)`.
    Adds an aggregate ``f"{phase}/the_metric"`` (sum of the non-loss
    metric values) — the scalar legacy callers used for checkpoint
    selection.
    """
    if task not in Evaluator._REGISTRY:
        raise ValueError(
            f"No evaluator for task '{task}'. "
            f"Known tasks: {sorted(Evaluator._REGISTRY)}"
        )
    if device is None:
        device = model.device

    model.eval()
    model.current_tasks = [task]
    with torch.no_grad():
        for batch in dataloader:
            batch = _to_device(batch, device)
            model._log_buffer = {}
            model(batch)

    metrics = model.epoch_metrics(phase)
    the_metric = sum(
        v for k, v in metrics.items() if not k.endswith("loss_epoch")
    )
    metrics[f"{phase}/the_metric"] = the_metric
    return metrics


def main(argv=None):
    """python -m renaissance.eval --checkpoint <path> --task <task> ..."""
    parser = argparse.ArgumentParser(
        description="Evaluate a RenaissanceModel checkpoint on a downstream task."
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

    from .modeling import RenaissanceModel

    print(f"Loading model from {args.checkpoint!r} ...")
    model = RenaissanceModel.from_pretrained(args.checkpoint)
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

    Thin dispatch layer; each dataset module owns its collate logic.
    See docs/data-preparation.md for dataset preparation instructions.
    """
    raise NotImplementedError(
        "Automatic dataloader construction is not yet implemented. "
        "Build the dataloader manually and call evaluate(model, dl, task) directly. "
        "See docs/data-preparation.md for dataset preparation instructions."
    )


if __name__ == "__main__":
    main()
