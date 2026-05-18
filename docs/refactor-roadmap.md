# Renaissance Refactor Roadmap

Steps 1–8 are complete; `v1.2.0` is tagged and `renaissance-1.2` is the stable default branch. Active development is on `renaissance-1.3-dev`. A post-1.2 hardening & cleanup pass (2026-05-17) is recorded below. Steps 9–10 remain.

---

## Dependency order summary

```
[1] Smoke tests          ✓ done
        ↓
[2] HF Datasets          ✓ done
        ↓
[3] HF Accelerate        ✓ done
        ↓
[4] New Interface        ✓ done
        ↓
[5] Model Hub            ✓ done
        ↓
[6] Eval & Benchmarking  ✓ done
        ↓
[7] Docs & Release       ✓ done (v1.2.0 tagged)
        ↓
[8] CI/CD Pipeline       ✓ partial (test + lint jobs live)
        ↓
[*] Post-1.2 hardening   ✓ done (2026-05-17)
        ↓
[9] Performance & Efficiency Optimizations  ← next
        ↓
[10] Extended Architecture Support
```

Each step keeps the smoke tests green as a regression guard.

---

## ✓ Step 1 — Smoke-Test Harness

**Delivered:** `tests/conftest.py`, `tests/test_smoke.py`, `tests/test_datasets.py`, `tests/test_trainer.py`. `test_trainer.py` was restored in the 2026-05-17 hardening pass (see below) and now runs in CI. `test_datasets.py` remains ignored — the legacy data layer imports `transformers.data`, which pulls in tensorflow and hits the `np.object` removal (numpy 1.20+); it comes back online when the legacy data layer is removed.

---

## ✓ Step 2 — Data Layer: PyArrow → HF Datasets

**Delivered:** All `*_dataset.py` files now wrap Arrow tables in `datasets.Dataset`; `write_*.py` scripts moved to `renaissance/utils/legacy/`.

---

## ✓ Step 3 — Training Framework: Lightning → HF Accelerate

**Delivered:** `RenaissanceTransformer` is a plain `nn.Module`; `renaissance/trainer.py` (`RenaissanceTrainer`) owns the Accelerate-based training loop; `renaissance_utils.set_schedule()` returns `(optimizer, scheduler)` directly; `epoch_wrapup()` returns a metrics dict.

---

## ✓ Step 4 — New Interface System (Sacred → omegaconf)

**Delivered:** `renaissance/config_schema.py` (5 typed dataclasses + `to_flat_dict` + `from_omegaconf`); new `run.py` (omegaconf YAML load + CLI dotlist overrides); `configs/` directory with 4 YAML files; `renaissance/config.py` preserved as `renaissance/config_legacy.py` (later deleted in the 1.3 line — see git history).

---

## ✓ Step 5 — Model Hub Integration

**Delivered:**
- `renaissance/hub.py` — `RenaissanceHubConfig(PretrainedConfig)` + `push_to_hub()`
- `save_pretrained(path)` / `from_pretrained(path_or_repo_id)` on `RenaissanceTransformer` — writes `config.json` + `model.safetensors`
- `save_checkpoint` in `RenaissanceTrainer` writes `model.safetensors` + `training_state/` (Accelerate optimizer/scheduler/RNG state)
- `tests/test_hub.py` — 9 tests (config round-trips, file creation, parameter + forward-output equality), all green

**Design decision:** Two-tower encoder sub-models (image encoder, text encoder) are re-downloaded from their HF repos at `from_pretrained` init time; only the fused weights are stored in `model.safetensors`. This keeps checkpoint size small at the cost of an internet connection on first load.

---

## ✓ Step 6 — Evaluation & Benchmarking Suite

**Delivered:**
- `renaissance/eval.py` — `Evaluator` base with `__init_subclass__` auto-registration; per-task subclasses: `MLMEvaluator`, `ITMEvaluator`, `VQAEvaluator`, `NLVR2Evaluator`, `SNLIEvaluator`, `RefEvaluator`, `Ref2Evaluator`, `IRTREvaluator`, `MRPCEvaluator`; `evaluate()` convenience function; CLI (`python -m renaissance.eval --checkpoint <path> --task <task> --split <split> --data_root <path>`)
- `tests/test_eval.py` — 16 deterministic tests (zero-weight models → known logits → asserted metric values), all green
- `docs/benchmarks.md` — placeholder template with column headers; fill in after training on real data

**Known limitation:** `IRTREvaluator` accumulates batch-level loss only. Full recall@k requires a two-pass full-corpus similarity sweep (see `compute_irtr_recall` in `objectives.py`) and a custom dataloader — not wired into the standard `Evaluator.__call__` loop.

---

## ✓ Step 7 — Documentation & Release

**Delivered:**
- `README.md` rewritten: 5-command quickstart, architecture overview, docs table, checkpoint format, eval CLI usage
- `docs/data-preparation.md` (replaces `DATA.md`): 8 datasets with download links, directory layouts, `make_arrow()` snippets
- `docs/configuration.md`: all 5 config groups tabulated (experiment, model, task, data, training)
- `docs/training.md`: gradient accumulation formula, distributed training, mixed precision, resume vs finetune, LR schedule, LR groups, tips
- `examples/pretrain_two_tower.ipynb`: end-to-end walkthrough using `SyntheticDataloader` (no downloads required)

**Pending:** `v1.1.0` tag — will be created on `main` after PR #1 is merged and the full test suite passes.

---

## ✓ Post-1.2 Hardening & Cleanup Pass (2026-05-17)

A fresh-eyes inspection of the post-rewrite tree surfaced stale docs, dead
code, an untested core component, and a half-finished data-layer migration.
All seven items landed on `renaissance-1.3-dev` (pushed to origin):

1. **README correctness** (`74bc2bf`) — fixed quickstart/checkpoint snippets
   that imported the deleted `renaissance.modules.RenaissanceTransformer` and
   the pre-relocation `utils.write_*` path; dropped `irtr` from the eval task
   list.
2. **Dead-code removal** (`74bc2bf`) — deleted `renaissance/config_legacy.py`
   (10.5k lines, imported `sacred` which is not a dependency, referenced
   nowhere); cleaned its ruff-exclude / CLAUDE.md / roadmap pointers.
3. **Trainer test coverage** (`d6fd4b5`) — `test_trainer.py` was ignored on a
   misdiagnosed excuse; rewrote it against `RenaissanceModel` + conftest
   fixtures and made the TensorBoard `SummaryWriter` import lazy + graceful
   (a logging backend must never crash training). +3 tests now in CI.
4. **Data-backend cutover** (`563f666`) — flipped the schema default
   `backend: legacy → modern`; pinned `backend: legacy` explicitly in the 4
   shipped configs so existing Arrow-on-disk runs are unchanged; added a
   `DeprecationWarning` to `MTDataModule`; documented migration. Legacy stack
   intact (additive-then-cutover); deletion deferred to a later phase.
5. **Doc consolidation** (`75270c6`) — deleted obsolete root `DATA.md` /
   `CONFIGURING_MODELS.md` duplicates; purged all stale
   `RenaissanceTransformer` references repo-wide.
6. **Packaging** (`3fef34f`) — `setup.py` now single-sources deps from
   `requirements.txt`; version bumped `1.2.0.dev0 → 1.3.0.dev0`.
7. **Housekeeping** (`9acbe6a` + git ops) — hardened the trainer loss-sum
   (`endswith("_loss")` vs substring match); pruned 7 fully-merged branches;
   archived 6 unmerged/no-remote branches as `archive/*` tags (pushed) before
   deletion; removed root scratch notebooks.

Suite went 181 → **184 passed / 1 skipped**, ruff clean throughout.

---

## Step 8 — CI/CD Pipeline (partial)

**Goal:** Run the full test suite automatically on every push and pull request so regressions are caught before merging, without requiring a GPU.

**Delivered:** `.github/workflows/ci.yml` with a `test` job (CPU-only PyTorch wheel + `pytest tests/`) and a `lint` job (`ruff check .`), triggered on push/PR against `renaissance-1.2` and `renaissance-1.3-dev`.

**Remaining tasks:**

1. Add `ruff format --check .` to the `lint` job.
2. Add a `type-check` job: `mypy renaissance/` with a minimal `mypy.ini` (strict on new files, lenient on legacy).
3. Cache the HuggingFace model config downloads (`.cache/huggingface`) across CI runs so `AutoConfig.from_pretrained` doesn't re-fetch every run.
4. Add a `docs-build` job that validates all links in `docs/` are not broken (`markdown-link-check` or similar).

**Watch out for:** The smoke tests download small HF configs (~few KB) at fixture time — ensure the CI runner has outbound internet or pre-cache the configs in the repo under `tests/fixtures/`.

---

## Step 9 — Performance & Efficiency Optimizations

**Goal:** Make training and inference faster and more memory-efficient without changing model behavior, enabling larger batch sizes and longer sequences on the same hardware.

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

**Tasks:**

1. **LLM text encoders:** Extend `TwoTowerEncoder` to support decoder-only models (LLaMA, Mistral, Phi) as the text tower by mean-pooling the last hidden state instead of using a CLS token. Gate on `text_encoder_pooling: "cls" | "mean" | "last"` config key.
2. **CLIP vision encoders:** Add support for `openai/clip-vit-*` image encoders; handle the CLIP-specific `CLIPVisionModel` output format in the projection layer.
3. **Contrastive pretraining (CLIP-style):** Implement `compute_contras` for symmetric InfoNCE loss as an alternative or complement to ITM. The `contras` loss key already exists in `loss_names` — wire it up.
4. **Cross-modal architecture variants:** Add a `cross_modal_fusion: "lxmert" | "co-attention" | "concat"` config option for experimenting with simpler fusion strategies alongside the existing LxmertXLayer approach.
5. Write architecture integration tests for each new encoder type that verify output shapes and finite losses end-to-end using the smoke-test pattern.

**Watch out for:** Decoder-only LLMs use causal attention masks; passing bidirectional text through them requires either removing the causal mask or using an encoder-only wrapper. Make the distinction explicit in config and documentation.
