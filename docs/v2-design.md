# Renaissance 2.0 — Ground-Up Redesign

**Status:** design draft (2026-07-22). Approved direction; prototyping next.
**Supersedes:** Step 10 of `refactor-roadmap.md` (its wishlist — LLM text encoders,
CLIP vision encoders, contrastive pretraining, alternative fusion — becomes routine
registry additions under this design). The queued NLVR2 regression was cancelled in
favor of reimplementing the task cleanly here.

---

## Design priorities

1. **Modularity** — adding a task, encoder, or fusion strategy should touch one file.
2. **Scalability** — large-model training (FSDP/DeepSpeed, streaming data) must be a
   config choice, never a code change.
3. **Ease of use** — usable without being an expert programmer or data scientist:
   recipes, a real CLI, validation that fails fast with actionable messages.
4. **Research-grade experiment tracking** — every run fully reproducible from a
   machine-readable manifest; eval is a first-class, versioned operation.
5. **HF-Hub-native** — models, tokenizers, and image processors load from and save to
   the Hub with standard `from_pretrained` / `push_to_hub` semantics.

## What we keep vs. discard

**Keep (proven in the 1.3 rewrite):**
- The registry *idea* (1.3's `TASK_REGISTRY`) — generalized to every component type.
- `hf_loader.py`'s centralization of `AutoModel`/`AutoConfig` loading, dim overrides,
  hidden-size detection.
- Task forward logic (loss computation per task) — ported, not redesigned.
- The thin Accelerate trainer loop as a starting skeleton.
- Step 9 perf flags (bf16, `gradient_checkpointing`, `use_flash_attention`).

**Discard:**
- `model_type = one-tower | two-tower` enum and the duplicated encoder implementations
  behind it.
- Config flattening (`to_flat_dict`) and the monolithic `config_schema.py`.
- The NLVR2 dual-image trick + `adjust_type_embeds_for_nlvr2` type-embed surgery.
- The legacy Arrow/Lightning data backend (already deprecated).
- Bespoke `.ckpt` serialization and rename-rule checkpoint archaeology.
- TensorBoard-only logging and `hparams.yaml`-based run reconstruction.

---

## 1. One registry mechanism for everything

A single decorator-based registry covers **encoders, fusion modules, heads, tasks,
datasets, trackers**. Each registered component contributes:

- a **name** (what configs reference),
- its **own config dataclass** (composed into the global schema — see §4),
- a **build function**.

```python
# registry.py (sketch)
@register("fusion", "cross-attention")
class CrossAttentionFusion(FusionModule):
    Config = CrossAttentionConfig          # component-owned schema
    def __init__(self, cfg: Config, encoders: EncoderSet): ...
```

Adding a component = one file, one decorator. No central wiring to edit. Third-party
extensions can register via Python entry points without forking the repo.

## 2. Composed model: `encoders + fusion + heads`

The model is a composition, not a class hierarchy:

```yaml
model:
  encoders:
    image: {hub_id: facebook/dino-vits16}
    text:  {hub_id: google/electra-small-discriminator}
  fusion:
    name: cross-attention       # or merged-attention, contrastive, ...
    num_layers: 6
    hidden_size: 384
```

- **Two-tower** ≡ two encoders + `cross-attention` fusion.
- **One-tower** ≡ one shared encoder + `merged-attention` fusion.
- Step-10 wishlist items are registry entries: decoder-only LLM text encoders
  (mean-pool adapter), CLIP vision encoders, `contrastive` fusion (InfoNCE).

**Multi-image is native.** Batches carry an image *set* (`images: [N, K, C, H, W]`);
fusion modules receive per-image token streams and a task-declared `n_images`. NLVR2
is reimplemented as an ordinary 2-image classification task — no type-embed
monkey-patch, no weight surgery.

Fusion modules expose the same uniform `EncoderOutput`-style contract as 1.3
(`pooled` / `text_tokens` / `image_tokens`) so heads stay generic.

## 3. Vertical task slices

**The 1.3 pain:** adding a task touches `tasks/`, head wiring, `config_schema.py`,
and `data/collate.py`. **The 2.0 rule:** a task is one self-contained module that
declares everything:

```python
@register("task", "nlvr2")
class Nlvr2Task(Task):
    Config = Nlvr2Config                   # task-owned config schema
    batch_spec = BatchSpec(n_images=2, text=True, label="binary")
    def build_head(self, fusion_dim): ...  # -> LinearClsHead(fusion_dim*2, 2)
    def collate(self, examples, processors): ...
    def forward(self, model, batch) -> TaskOutput: ...
    def metrics(self) -> list[str]: ...    # ["accuracy"]
```

- `batch_spec` drives the data layer: the loader knows what fields to produce and
  how many images to decode, without task-specific branches in `collate.py`.
- Preprocessing uses the checkpoint's own processors (§6), passed in — tasks never
  hardcode transforms.
- Metrics remain declared-by-task, built centrally (keep 1.3's `TaskMetrics` shape).

## 4. Typed, composed config — never flattened

- **Nested typed config end-to-end** (dataclasses via omegaconf structured configs).
  `RenaissanceModel` and every component receive their typed sub-config, not a flat
  dict. `to_flat_dict` does not exist in 2.0.
- **Schema composition:** the global schema is assembled from the registries — the
  config keys under `model.fusion` are whatever the selected fusion's `Config`
  declares. Validation errors are precise and early:
  `model.fusion.num_layers: expected int, got 'twelve'`.
- **Recipes:** named, composable presets shipped in `recipes/`
  (`pretrain-small.yaml`, `finetune-nlvr2.yaml`). A user config is ~10 lines of
  overrides on a recipe; experts can override any leaf from the CLI with the same
  dotted paths as today.

## 5. Data layer: streaming-first, task-driven

- Single backend, HF `datasets` Hub-first (the 1.3 `modern` backend's direction),
  with **streaming as the default for pretraining-scale corpora** — no requirement
  that datasets fit on disk. Map-style remains available for small finetune sets.
- The loader is generic: it reads each active task's `batch_spec`, applies the
  checkpoint-derived processors, and delegates example→tensor conversion to the
  task's `collate`. No per-dataset datamodule classes.
- Multi-task sampling (mixing ratios, temperature sampling) is a loader-level config
  concern, not baked into datasets.

## 6. HF-Hub-native serialization (biggest single change)

- The composed model implements **`PyTorchModelHubMixin`** (or subclasses
  `PreTrainedModel`): `save_pretrained` / `from_pretrained` / `push_to_hub` work out
  of the box. Checkpoints are **safetensors + config.json**, loadable by anyone,
  from local disk or the Hub, with zero rename rules.
- The saved config.json *is* the resolved model config — reconstruction of a model
  from a checkpoint never involves parsing `hparams.yaml`.
- **Processors travel with checkpoints:** tokenizer from the text encoder's
  checkpoint, `AutoImageProcessor` from the vision encoder's checkpoint. This
  deletes `train_transform_keys` / `val_transform_keys` and the entire
  transform-mismatch bug class. (Custom augmentation like RandAug remains available
  as explicit, opt-in train-time config.)

## 7. Trainer & scalability

- Keep a **thin loop on Accelerate**, but with hard rules:
  - No single-process assumptions anywhere; metrics via torchmetrics with proper
    cross-rank sync.
  - **FSDP2 / DeepSpeed selected purely by config** (`training.distributed.strategy`),
    with per-encoder FSDP auto-wrap policies contributed by encoder registry entries.
  - **Sharded, distributed-aware checkpointing** (safetensors), resumable mid-epoch
    for streaming datasets.
  - bf16 default on supporting hardware; `gradient_checkpointing`,
    `use_flash_attention`, `torch.compile` as first-class flags.
- Grad accumulation stays auto-computed from
  `batch_size / (per_gpu_batchsize × world_size)`.

## 8. Experiment tracking & eval

- **Pluggable tracker interface**; W&B as the recommended default, TensorBoard and
  MLflow as drop-ins. Trackers receive structured events (step losses, epoch
  metrics, throughput), not ad-hoc `log()` calls scattered through tasks.
- **Run manifest** (`manifest.json`, written at launch and finalized at exit):
  fully-resolved config, git SHA + dirty-diff, package versions, hardware,
  dataset fingerprints (Hub revision hashes), wall-clock, final metrics, checkpoint
  paths. *Re-running an experiment is `rena train --from-manifest <path>`.*
- **Eval is a first-class command** producing a versioned eval report stored next to
  the checkpoint. Regression testing becomes
  `rena eval <ckpt> --against <old-report>` — no more bespoke
  `scripts/regress_*_ckpt.py` per task.

## 9. CLI & UX

```bash
rena train  recipe=pretrain-small encoders.image.hub_id=facebook/dino-vits16
rena eval   <ckpt-or-hub-id> data=nlvr2
rena predict <ckpt-or-hub-id> --input image.jpg --text "..."
rena push   <ckpt> username/model-name
rena train ... --dry-run    # build model + 1 batch + 1 fwd/bwd,
                            # print param counts & memory estimate, exit
```

- Subcommand CLI (tyro or typer) replaces `run.py <yaml> key=value`.
- `--dry-run` catches most config mistakes in ~30 seconds.
- Errors are actionable: bad Hub id → "checkpoint not found, did you mean …";
  OOM in dry-run → suggests `per_gpu_batchsize` / checkpointing flags.

---

## Package layout

```
renaissance/
  registry.py          # one registry mechanism for everything
  components/
    encoders/          # @register("encoder", ...) — hf_loader lives here
    fusion/            # cross_attn.py, merged_attn.py, contrastive.py
    heads/             # Pooler, MlmHead, ItmHead, LinearClsHead
  tasks/               # one file per task: config + batch_spec + collate
                       #   + head + loss + metrics  (vertical slices)
  data/                # generic streaming loader; multi-task mixing
  trainer/             # thin Accelerate loop; fsdp/deepspeed via config
  tracking/            # tracker interface, manifest writer, eval reports
  model.py             # composed model + PyTorchModelHubMixin
  config.py            # schema composition from registries; recipes
  cli.py               # rena train|eval|predict|push
recipes/               # named preset configs
```

## Build order (prototype phases)

Phase-at-a-time, additive where possible; each phase lands with tests.

1. **Core skeleton** — `registry.py`, component protocols (`Encoder`, `FusionModule`,
   `Head`, `Task`), composed `model.py` with `PyTorchModelHubMixin`. Golden test:
   compose a two-tower (dino-vits16 + electra-small + cross-attention) and match
   1.3's forward output shapes.
2. **Config & CLI** — schema composition, recipes, `rena train --dry-run` against a
   toy config.
3. **Data layer** — generic loader honoring `batch_spec`, checkpoint-derived
   processors, streaming path.
4. **Two proving tasks** — `mlm` + `itm` ported as vertical slices; short pretrain
   smoke run.
5. **Trainer hardening** — FSDP/DeepSpeed config paths, sharded checkpoints,
   manifest writer, tracker interface.
6. **Task ports** — `snli`, `vqa`, `ref`/`ref2`, `mrpc`; **NLVR2 reimplemented** as a
   plain 2-image task (fresh design — no parity target against old checkpoints).
7. **Eval & reports** — `rena eval`, versioned reports, SNLI-VE quality check vs.
   1.3 numbers (quality sanity, not byte parity — new preprocessing means new
   numbers are expected).
8. **Cutover** — 2.0 becomes the mainline; 1.3 line frozen for reference.

## Open questions

- `PyTorchModelHubMixin` vs. full `PreTrainedModel` subclass? Mixin is lighter and
  sufficient for `save/from_pretrained` + `push_to_hub`; `PreTrainedModel` buys
  `AutoModel` registration and ecosystem integration at the cost of conforming to
  its constraints. Start with the mixin; revisit at Phase 5.
- Config engine: plain omegaconf structured configs vs. full Hydra. Leaning
  omegaconf-only (we don't need Hydra's launcher/sweep machinery; W&B sweeps can
  cover hyperparameter search).
- W&B as default tracker requires an account — keep TensorBoard as the zero-setup
  fallback for offline users.
- Whether 2.0 lives on a new branch off `renaissance-1.3-dev` (`renaissance-2.0-dev`)
  or a fresh top-level package developed in-tree alongside 1.3. Decide at Phase 1.
