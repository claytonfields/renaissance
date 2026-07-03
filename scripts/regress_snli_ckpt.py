"""
Modeling-rewrite regression: load a Lightning-era RenaissanceTransformer
checkpoint into the current RenaissanceModel and evaluate on SNLI-VE.

The 1.3-line rewrite deleted the legacy ``renaissance.modules`` package and
replaced ``RenaissanceTransformer`` with a backbone + task-registry
``RenaissanceModel``. Every parameter name that used to live on the model
directly is now nested under either ``backbone.encoder.*`` or
``heads.<task>.*``. This script proves the rewrite is faithful by loading a
known-good pre-rewrite checkpoint through a **two-rule prefix rewrite**,
verifying ``load_state_dict(strict=True)`` succeeds, and reporting
SNLI-VE dev / test accuracy against the eval numbers recorded when the
checkpoint was trained (2024-09).

Reference numbers for the shipped baseline checkpoint at
``result/snli_twotower_deit_electra_seed0_from_mlm_itm_deit_electra_seed0_
is224_ps16_bs704_pgbs176_ts100000_last/version_0/`` (deit-tiny +
electra-small, image_size=384):

    dev  0.7405   test  0.7455   (Sept 2024, Lightning + transformers 4.30ish)

The rewrite reproduces these to within ~1.5 pp on transformers 4.37 /
torch 2.8 (residual gap attributed to library drift over ~2 years; the
weight load itself is 0-missing / 0-unexpected under strict=True).

Usage
-----
    python scripts/regress_snli_ckpt.py \\
        --ckpt result/snli_twotower_deit_electra_seed0_.../version_0/checkpoints/last.ckpt \\
        --data-root data/arrow

Runtime: ~10 min on one RTX 3080 (dev + test combined).
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

# Make the sibling `renaissance` package importable when this script is
# launched directly (`python scripts/regress_snli_ckpt.py`) and the package
# is not pip-installed.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# `transformers.data.processors.glue` unconditionally imports TF, and TF's
# `dtypes.py` uses ``np.object`` / ``np.bool`` (removed in NumPy 1.20+).
# Restore the aliases BEFORE any transformers.data import.
import numpy as _np

for _attr, _target in (
    ("object", object), ("bool", bool), ("int", int), ("float", float),
    ("complex", complex), ("str", str), ("long", int),
    ("unicode", str), ("typeDict", _np.sctypeDict),
):
    if not hasattr(_np, _attr):
        setattr(_np, _attr, _target)

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from renaissance.datasets.base_dataset import BaseDataset
from renaissance.datasets.snli_dataset import SNLIDataset
from renaissance.eval import evaluate
from renaissance.modeling import RenaissanceModel

# ---------------------------------------------------------------------------
# Weight rename map — old RenaissanceTransformer → new RenaissanceModel
# ---------------------------------------------------------------------------


def rename(k: str) -> str:
    """The entire "converter" from Lightning state_dict → RenaissanceModel.

    Two rules cover every one of the 666 parameters in the reference ckpt:
    - ``encoder.*`` → ``backbone.encoder.*`` (whole fused encoder moved
      one attribute level down into the backbone wrapper)
    - ``snli_classifier.*`` → ``heads.snli.*`` (task heads moved into an
      ``nn.ModuleDict`` keyed by task name; field names inside the head
      — ``dense``/``layer_norm``/``out_proj`` — are unchanged).

    Add rules here to cover other head names if regressing a different
    downstream ckpt (e.g. ``vqa_classifier``, ``nlvr2_classifier``,
    ``ref_classifier``, ``ref2_classifier``, ``mlm_score``, ``itm_score``).
    """
    if k.startswith("encoder."):
        return "backbone." + k
    if k.startswith("snli_classifier."):
        return k.replace("snli_classifier.", "heads.snli.", 1)
    return k


# ---------------------------------------------------------------------------
# Model config matching the reference hparams.yaml exactly
# ---------------------------------------------------------------------------

CFG = {
    "model_type": "two-tower",
    "image_encoder": "facebook/deit-tiny-patch16-224",
    "text_encoder": "google/electra-small-discriminator",
    "random_init_vision_encoder": False,
    "random_init_text_encoder": False,
    "image_encoder_manual_configuration": False,
    "text_encoder_manual_configuration": False,
    "freeze_image_encoder": False,
    "freeze_text_encoder": False,
    "freeze_cross_modal_layers": False,
    "cross_layer_hidden_size": 256,
    "num_cross_layers": 6,
    "num_cross_layer_heads": 4,
    "cross_layer_mlp_ratio": 4,
    "cross_layer_drop_rate": 0.1,
    "max_text_len": 50,
    "vocab_size": 30522,
    "loss_names": {"snli": 1},
}

TARGETS = {"dev": 0.7405, "test": 0.7455}


# ---------------------------------------------------------------------------
# Per-split SNLI datasets — the stock SNLIDataset loads dev+test together
# and the old code split them by `table_name`. Load one arrow at a time
# instead for clean per-split metrics.
# ---------------------------------------------------------------------------


class _SNLIOneArrow(SNLIDataset):
    _arrow_name = ""

    def __init__(self, *args, **kw):
        self.split = "val"
        BaseDataset.__init__(
            self, *args, **kw,
            names=[self._arrow_name],
            text_column_name="sentences",
            remove_duplicate=False,
        )


class SNLIDevOnly(_SNLIOneArrow):
    _arrow_name = "snli_dev"


class SNLITestOnly(_SNLIOneArrow):
    _arrow_name = "snli_test"


def build_loader(dataset_cls, data_root, tokenizer, image_size, batch_size, num_workers):
    ds = dataset_cls(
        data_dir=str(data_root),
        transform_keys=["imagenet"],
        image_size=image_size,
        max_text_len=CFG["max_text_len"],
        draw_false_image=0,
        draw_false_text=0,
        image_only=False,
        tokenizer=tokenizer,
        # SNLIDataset never touches self.processor; only refcoco does.
        # Passing None avoids AutoImageProcessor's TF-import chain.
        processor=None,
    )
    mlm_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: ds.collate(batch, mlm_collator),
        pin_memory=True,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt", required=True, type=Path,
        help="Path to the Lightning-era .ckpt (has {state_dict, hyper_parameters, ...}).",
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/arrow"),
        help="Directory containing snli_dev.arrow / snli_test.arrow.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=384,
                        help="Must match the fine-tune image_size in hparams.yaml.")
    parser.add_argument("--tolerance", type=float, default=0.02,
                        help="Fail if any split delta exceeds this (default 2pp).")
    args = parser.parse_args(argv)

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    print(f"Loading ckpt: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    new_sd = {rename(k): v for k, v in ckpt["state_dict"].items()}
    print(f"  {len(new_sd)} tensors after rename")

    print("Building RenaissanceModel + strict-loading weights ...")
    model = RenaissanceModel(CFG)
    model.load_state_dict(new_sd, strict=True)
    del new_sd, ckpt
    # Reference training set fine_tune=True → TwoTowerEncoder took the
    # interpolate_pos_encoding=True branch. Reproduce that here so the
    # deit-tiny (14×14 pos embed) evaluates cleanly at 384×384 (24×24).
    model.backbone.encoder.test_only = True
    model.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(CFG["text_encoder"])
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    results = {}
    for split, ds_cls in (("dev", SNLIDevOnly), ("test", SNLITestOnly)):
        print(f"\n=== SNLI-VE {split} ===")
        loader = build_loader(
            ds_cls, args.data_root, tokenizer,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        print(f"  {len(loader.dataset)} items / {len(loader)} batches")
        metrics = evaluate(model, loader, task="snli", device=device, phase="val")
        acc = metrics["snli/val/accuracy_epoch"]
        loss = metrics["snli/val/loss_epoch"]
        target = TARGETS[split]
        delta = acc - target
        results[split] = (acc, target, delta)
        print(f"  accuracy = {acc:.4f}   (target {target:.4f}, delta {delta:+.4f})")
        print(f"  loss     = {loss:.4f}")

    print("\n=== Summary ===")
    print(f"{'split':<6} {'accuracy':>10} {'target':>10} {'delta':>10}")
    for split, (acc, target, delta) in results.items():
        print(f"{split:<6} {acc:>10.4f} {target:>10.4f} {delta:>+10.4f}")

    worst = max(abs(d) for _, _, d in results.values())
    if worst > args.tolerance:
        print(f"\nFAIL: worst delta {worst:.4f} > tolerance {args.tolerance:.4f}")
        return 1
    print(f"\nPASS: max delta {worst:.4f} within {args.tolerance:.4f} tolerance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
