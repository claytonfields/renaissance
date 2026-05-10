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
[7] Documentation & Release (v1.1.0 tag)
        ↓
[8] CI/CD Pipeline
        ↓
[9] Performance & Efficiency Optimizations
        ↓
[10] Extended Architecture Support
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

---

## Step 8 — CI/CD Pipeline

**Goal:** Run the full test suite automatically on every push and pull request so regressions are caught before merging, without requiring a GPU.

**Why eighth:** CI is only worth setting up once the test suite (Steps 1, 2, 3) and the package structure (Steps 4–7) are stable. Adding CI to a moving target means constant workflow breakage.

**Tasks:**

1. Add `.github/workflows/ci.yml` that runs `pytest tests/` on every push and PR against `main` and `dev-interface`.
2. Pin a CPU-only PyTorch wheel in the CI environment to keep runner time under 5 minutes.
3. Add a `lint` job: `ruff check .` + `ruff format --check .` (or `black` + `flake8` if already in use).
4. Add a `type-check` job: `mypy renaissance/` with a minimal `mypy.ini` (strict on new files, lenient on legacy).
5. Cache the HuggingFace model config downloads (`.cache/huggingface`) across CI runs so `AutoConfig.from_pretrained` doesn't re-fetch every run.
6. Add a `docs-build` job that validates all links in `docs/` are not broken (`markdown-link-check` or similar).

**Watch out for:** The smoke tests download small HF configs (~few KB) at fixture time — ensure the CI runner has outbound internet or pre-cache the configs in the repo under `tests/fixtures/`.

---

## Step 9 — Performance & Efficiency Optimizations

**Goal:** Make training and inference faster and more memory-efficient without changing model behavior, enabling larger batch sizes and longer sequences on the same hardware.

**Why ninth:** Optimizations are safest to apply after the architecture and training loop are locked in (Steps 2–4) and the eval harness (Step 6) exists to verify that numbers don't regress. Applying them earlier risks re-doing work if the underlying code changes.

**Tasks:**

1. **Flash Attention:** Replace standard `nn.MultiheadAttention` / HF attention layers with `flash_attn` where supported. Gate behind a config flag (`use_flash_attention: bool`) so the fallback path stays testable on CPU.
2. **Gradient checkpointing:** Enable `model.gradient_checkpointing_enable()` for the text and image encoder towers; expose as a `gradient_checkpointing: bool` config key.
3. **`torch.compile`:** Wrap the model with `torch.compile(model, mode="reduce-overhead")` behind a flag; measure throughput delta and document in `docs/benchmarks.md`.
4. **Parameter-efficient fine-tuning (LoRA/adapters):** Integrate `peft` library to allow fine-tuning with LoRA adapters on the encoder towers. Expose `use_lora: bool`, `lora_r`, `lora_alpha` config keys.
5. **Mixed precision audit:** Confirm `bfloat16` works end-to-end (not just `float16`) and add a CI smoke run with `mixed_precision="bf16"`.
6. Benchmark each optimization in isolation on a standard config and document the throughput / memory numbers in `docs/benchmarks.md`.

**Watch out for:** Flash Attention requires CUDA compute capability ≥ 8.0 (Ampere+). The fallback path must remain correct and tested on older GPUs and CPU. LoRA adapters interact with the Hub integration (Step 5) — `save_pretrained` must serialize adapter weights separately from base weights.

---

## Step 10 — Extended Architecture Support

**Goal:** Broaden the set of supported encoder backbones to include modern LLM-scale text encoders and CLIP-family vision encoders, enabling experiments at a larger scale than the original METER-era models.

**Why tenth:** Architecture extensions are additive — they slot into the existing two-tower encoder pattern without changing the training loop, data layer, or interface. Doing this last means the full infrastructure (CI, Hub integration, eval harness) is in place to validate new architectures immediately.

**Tasks:**

1. **LLM text encoders:** Extend `TwoTowerEncoder` to support decoder-only models (LLaMA, Mistral, Phi) as the text tower by mean-pooling the last hidden state instead of using a CLS token. Gate on `text_encoder_pooling: "cls" | "mean" | "last"` config key.
2. **CLIP vision encoders:** Add support for `openai/clip-vit-*` image encoders; handle the CLIP-specific `CLIPVisionModel` output format in the projection layer.
3. **Contrastive pretraining (CLIP-style):** Implement `compute_contras` for symmetric InfoNCE loss as an alternative or complement to ITM. The `contras` loss key already exists in `loss_names` — wire it up.
4. **Cross-modal architecture variants:** Add a `cross_modal_fusion: "lxmert" | "co-attention" | "concat"` config option for experimenting with simpler fusion strategies alongside the existing LxmertXLayer approach.
5. Write architecture integration tests for each new encoder type that verify output shapes and finite losses end-to-end using the smoke-test pattern.

**Watch out for:** Decoder-only LLMs use causal attention masks; passing bidirectional text through them requires either removing the causal mask or using an encoder-only wrapper. Make the distinction explicit in config and documentation.
