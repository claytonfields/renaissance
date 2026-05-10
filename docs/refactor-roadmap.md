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
```

Each step keeps the smoke tests green as a regression guard. Add integration tests at the end of Steps 2 and 3 before moving on.
