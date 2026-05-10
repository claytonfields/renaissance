# Renaissance Refactor Roadmap

Step 1 (smoke-test harness) is complete. The remaining steps are ordered to minimize breakage: each step builds on a stable foundation before the next layer is replaced.

---

## Step 2 — Data Layer: PyArrow → HF Datasets

**Goal:** Replace all `pa.ipc.RecordBatchFileReader` Arrow loading with HuggingFace `datasets.Dataset`, and retire the `write_*.py` serialization scripts in favor of standard dataset loading/caching.

**Why second:** The data layer is the most self-contained layer. Swapping it out does not require changing the model or training loop — only the `*_dataset.py` files and `multitask_datamodule.py`. Once datasets load correctly, the smoke tests (which use synthetic data) continue to pass, and you have a clean integration point before touching the training framework.

**Tasks:**

1. Audit every `*_dataset.py` under `renaissance/datasets/` to inventory all Arrow-specific read paths.
2. Replace `pa.ipc.RecordBatchFileReader` with `datasets.load_from_disk` (for pre-converted datasets) or `datasets.load_dataset` (for datasets available on the Hub).
3. Replace `__getitem__` row fetches (`table.slice(idx, 1).to_pydict()`) with standard dict indexing.
4. Update `multitask_datamodule.py` to remove Arrow-specific path logic and accept HF `Dataset` objects.
5. Retire `renaissance/utils/write_*.py` scripts (or move to a `legacy/` subfolder for reference).
6. Add integration tests in `tests/test_datasets.py` that load a tiny in-memory HF dataset and verify the collation output shapes match what the model expects.

**Watch out for:** NLVR2 and RefCOCO have non-standard collation (paired images, region stacks) — verify those collate functions produce the same dict structure the objectives expect.

---

## Step 3 — Training Framework: Lightning → HF Accelerate

**Goal:** Replace `pl.LightningModule` / `pl.Trainer` with a manual training loop driven by HuggingFace `Accelerate`, keeping distributed training, mixed precision, and gradient accumulation.

**Why third:** Accelerate wraps the model, optimizer, and dataloaders — it needs the data layer (Step 2) to be stable so it can wrap real dataloaders. Doing this before Step 2 would mean rewriting the loop twice.

**Tasks:**

1. Extract all logic currently in `RenaissanceTransformer` that belongs to Lightning (`training_step`, `validation_step`, `test_step`, `configure_optimizers`, `log`) into a standalone `Trainer` class in `renaissance/trainer.py`.
2. Remove `pl.LightningModule` as the base class; make `RenaissanceTransformer` a plain `nn.Module`.
3. Wire up `accelerator = Accelerator(mixed_precision=..., gradient_accumulation_steps=...)` at the top of the training loop.
4. Replace `self.log(...)` calls in objectives with explicit metric accumulation (TorchMetrics or plain dicts passed into the trainer).
5. Replace Sacred's `@ex.automain` entry point in `run.py` with a simple `argparse` or `jsonargparse` CLI (see Step 4).
6. Port callbacks: checkpoint saving (use `accelerator.save_state`), LR scheduling, and TensorBoard/WandB logging.
7. Add a fast training smoke test: 2-step loop on synthetic data, assert loss decreases.

**Watch out for:** `set_task()` in `renaissance_utils.py` mutates `self.current_tasks` each step — make sure the new training loop calls this correctly for multi-task batches. DDP find-unused-parameters behavior must be explicitly configured in Accelerate via `kwargs_handlers`.

---

## Step 4 — New Interface System

**Goal:** Replace Sacred (`@ex.config`, `@ex.named_config`, `with key=value` CLI) with a clean, typed configuration system that is easy to use, introspectable, and does not require Sacred's observer infrastructure.

**Why last:** The interface sits on top of everything else. Changing it while the training framework and data layer are still in flux would require rewriting it multiple times. By doing it last, you can design around the final Accelerate-based trainer and HF Datasets API.

**Tasks:**

1. Choose a config backend. Recommended: **`jsonargparse`** (used by Lightning CLI) or **`dataclasses` + `omegaconf`**. Both support nested configs, CLI overrides, and YAML/JSON config files.
2. Define typed config dataclasses for each concern: `ModelConfig`, `TrainingConfig`, `DataConfig`, `TaskConfig`.
3. Write a new `run.py` entry point that: loads a YAML config file, applies CLI overrides, validates types, instantiates model + trainer + datamodule, and calls `trainer.fit()`.
4. Provide example YAML configs in `configs/` for each major use case (pretrain one-tower, pretrain two-tower, finetune NLVR2, finetune VQA, etc.).
5. Remove all Sacred decorators and `@ex.named_config` functions from `renaissance/config.py`; the file can be deleted or reduced to the loss-names registry only.
6. Update `CLAUDE.md` and `README.md` with the new CLI invocation syntax.

**Watch out for:** Sacred's `_config` dict is passed directly into `RenaissanceTransformer.__init__` and referenced throughout as `self.hparams`. After moving to dataclasses, ensure all downstream references to `self.hparams["key"]` are updated to attribute access on the typed config object.

---

## Dependency order summary

```
[1] Smoke tests (done)
        ↓
[2] HF Datasets
        ↓
[3] HF Accelerate
        ↓
[4] New Interface
        ↓
[5] Model Hub Integration
        ↓
[6] Evaluation & Benchmarking Suite
        ↓
[7] Documentation & Release
```

Each step keeps the smoke tests green as a regression guard. Add integration tests at the end of Steps 2 and 3 before moving on.

---

## Step 5 — Model Hub Integration

**Goal:** Make pretrained checkpoints loadable from the HuggingFace Hub via a standard `from_pretrained` pattern, and enable pushing fine-tuned models back to the Hub.

**Why fifth:** Depends on the Accelerate-based trainer (Step 3) for checkpoint format and the typed config system (Step 4) for serializing model configuration alongside weights. Both must be stable before a public checkpoint format is locked in.

**Tasks:**

1. Subclass `PreTrainedModel` (or implement the `push_to_hub` / `save_pretrained` / `from_pretrained` interface) in `RenaissanceTransformer`.
2. Define a `RenaissanceConfig` class that subclasses `PretrainedConfig` and serializes all model hyperparameters to `config.json`.
3. Implement `save_pretrained(path)` to write `config.json` + `pytorch_model.bin` (or sharded weights).
4. Implement `from_pretrained(repo_id_or_path)` that reconstructs the full model from config and loads weights.
5. Add a `push_to_hub` helper (wraps `huggingface_hub.HfApi`) for publishing checkpoints.
6. Write a test that round-trips: save → load → assert parameter equality and identical forward pass output.

**Watch out for:** The two-tower encoder loads sub-models from their own HF repos at init time. `from_pretrained` must either re-download those backbones or store their weights inline — decide on one strategy and document it.

---

## Step 6 — Evaluation & Benchmarking Suite

**Goal:** Provide a unified, reproducible evaluation harness for all downstream tasks, decoupled from training, that reports standard benchmark metrics (VQA accuracy, NLVR2 accuracy, SNLI-VE accuracy, IRTR R@1/R@5/R@10, RefCOCO accuracy@0.5 IoU).

**Why sixth:** Evaluation code currently lives inside `test_step` in the Lightning module and is tangled with training infrastructure. Once training and checkpointing are clean (Steps 3–5), the eval harness can be written as a standalone script that loads any checkpoint and evaluates it on a specified split.

**Tasks:**

1. Create `renaissance/eval.py` with an `evaluate(model, dataloader, task)` function that returns a metrics dict.
2. Port all `*_epoch_end` metric aggregation logic from `renaissance_module.py` into task-specific `Evaluator` classes.
3. Add a CLI entry point (`python -m renaissance.eval --checkpoint <path> --task vqa --split val --data_root <path>`) that prints results to stdout and optionally writes a JSON report.
4. Write deterministic eval tests using fixed synthetic batches that assert metric values match expected outputs (tests inputs → expected logits → expected metric).
5. Document expected numbers for each task on the standard val splits in `docs/benchmarks.md`.

**Watch out for:** IRTR evaluation requires computing similarity scores over the entire val set (not per-batch) — it needs to be implemented as a two-pass accumulation, not a streaming metric.

---

## Step 7 — Documentation & Release

**Goal:** Produce complete end-user documentation, a worked example notebook, and a clean `v1.1.0` release tag so the project is reproducible from scratch by someone new to the codebase.

**Why last:** Docs accurately reflect the final system. Writing them before Steps 2–6 are done means rewriting them as the interface changes.

**Tasks:**

1. Rewrite `README.md`: installation, quickstart (pretrain + finetune in 5 commands), link to `docs/`.
2. Write `docs/data-preparation.md` (replaces `DATA.md`) covering HF Datasets loading and any custom conversion still required.
3. Write `docs/configuration.md` documenting all config fields, defaults, and YAML examples.
4. Write `docs/training.md` covering distributed training, gradient accumulation, mixed precision, and resuming from checkpoints.
5. Create `examples/pretrain_two_tower.ipynb` — end-to-end walkthrough on a small dataset.
6. Tag `v1.1.0` on `main` after all prior steps are merged and the full test suite passes.

**Watch out for:** Keep example notebooks runnable on a single GPU with small synthetic or publicly available data so they can be executed in CI or by a reviewer without a large cluster.
