# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## Running Experiments

All training and evaluation is driven through `run.py` with a YAML config file. CLI `key=value` overrides use dotted paths matching the config groups below:

```bash
# Pretrain two-tower
python run.py configs/pretrain_two_tower.yaml \
  data.data_root=data/arrow/ training.num_gpus=1

# Pretrain one-tower
python run.py configs/pretrain_one_tower.yaml \
  model.encoder=facebook/dino-vits16 training.max_steps=50000

# Fine-tune
python run.py configs/finetune_nlvr2.yaml \
  experiment.load_path=<CKPT> model.image_size=288

# Test only
python run.py configs/finetune_vqa.yaml \
  experiment.load_path=<CKPT> experiment.test_only=true
```

Gradient accumulation is computed automatically: `grad_steps = batch_size / (per_gpu_batchsize × num_gpus × num_nodes)`.

Results and TensorBoard logs are written to `result/<exp_name>_seed<N>_is<img>_ps<patch>_bs<bs>_pgbs<pgbs>_ts<steps>/`.

## Configuration System

Config lives in typed dataclasses (`renaissance/config_schema.py`) and YAML files under `configs/`. The schema has five groups — `experiment`, `model`, `task`, `data`, `training` — which are flattened into a plain dict (via `to_flat_dict`/`from_omegaconf`, both of which run `normalize_tasks`) before being passed to `RenaissanceModel`.

Key config groups:
- **Model type**: `model.model_type = 'one-tower' | 'two-tower'`
- **One-tower**: `model.encoder`, `model.pooler_type` (`single` | `double`), `model.random_init_encoder`, dimension overrides
- **Two-tower**: `model.image_encoder`, `model.text_encoder`, `model.freeze_image_encoder`, `model.freeze_text_encoder`, cross-modal dims (`model.cross_layer_hidden_size`, `model.num_cross_layers`, `model.num_cross_layer_heads`, `model.cross_layer_mlp_ratio`, `model.cross_layer_drop_rate`)
- **Training**: `training.max_steps`, `training.max_epoch`, `data.batch_size`, `data.per_gpu_batchsize`, `training.learning_rate`, `training.warmup_steps`
- **Tasks / losses**: either `task.tasks` (a list, e.g. `["mlm", "itm"]`, preferred) or the legacy `task.loss_names` dict (set a task to `1`). `normalize_tasks` reconciles the two — a non-empty `tasks` list wins and rebuilds `loss_names`; otherwise `tasks` is derived from `loss_names > 0`. Registered tasks: `mlm`, `itm`, `vqa`, `nlvr2`, `snli`, `ref`, `ref2`, `mrpc`. (`irtr` and the stub GLUE tasks were dropped in the modeling rewrite.)

The legacy Sacred-based config is preserved at `renaissance/config_legacy.py`.

## Architecture Overview

### Entry point
`run.py` → loads YAML + CLI overrides via omegaconf → instantiates the data backend + `RenaissanceModel` → hands off to `RenaissanceTrainer` (Accelerate-based).

### Modeling layer (`renaissance/modeling/`)
The modeling stack was rewritten into a backbone-protocol + task-registry design (the legacy `renaissance/modules/renaissance_module.py` is gone).

- **`RenaissanceModel` (`model.py`)** — plain `nn.Module`. `__init__` is a loop building one head per active task into an `nn.ModuleDict`; `forward` is a loop over `self.current_tasks` running each task. Adding a task touches neither method. `set_active_tasks()` selects the configured tasks; metrics are owned by the model (`self.metrics`, a `TaskMetrics`) and flushed via `epoch_metrics(phase)`; step losses buffer in `self._log_buffer`.
- **Backbones (`backbones/`)** — `Backbone` protocol returning a uniform `EncoderOutput` (`pooled` / `text_tokens` / `image_tokens`, with legacy `cls_feats`/`text_feats`/`image_feats` aliases). `OneTowerBackbone` and `TwoTowerBackbone` currently wrap the legacy encoders in `renaissance/modules/`; `hf_loader.py` centralizes all `AutoModel`/`AutoConfig` loading + dim-override logic. (Encoder internals get modernized in the planned Phase 8.)
- **Heads (`heads.py`)** — `Pooler`, `MlmHead`, `ItmHead`, and one generic `LinearClsHead(in_dim, num_labels, hidden_dim=None, pool=False)` covering VQA/SNLI/ref/ref2/NLVR2/GLUE/CIFAR-10.
- **Tasks (`tasks/`)** — one `Task` per objective (`mlm`, `itm`, `vqa`, `nlvr2`, `snli`, `ref`, `ref2`, `mrpc`) in `TASK_REGISTRY`. Each declares `build_head(backbone, config)`, `metric_names()`, and `forward(model, batch) -> TaskOutput(loss, logits, targets, extras)`. No `pl_module` reach-back. NLVR2's dual-image trick is local to `Nlvr2Task`.
- **Metrics (`metrics.py`)** — `TaskMetrics` builds per-(phase, task, metric) objects from each task's `metric_names()` plus an always-on loss scalar; `compute(phase)` returns `{task/phase/metric_epoch: float}`.

`renaissance/modules/` still holds the wrapped legacy encoders (`one_tower_encoder`, `two_tower_encoder`, `embeddings`, `fusion_encoder`), `heads.Pooler`, `objectives.init_weights`, `dist_utils`, and `renaissance_utils.set_schedule`; it is deleted entirely in the planned Phase 8.

### Data pipeline
`renaissance/datamodules/multitask_datamodule.py` (`MTDataModule`) coordinates multiple `DataModule` instances. Each dataset has a corresponding `*_datamodule.py` and `*_dataset.py`. Datasets are pre-serialized to [Apache Arrow](https://arrow.apache.org/) format using the scripts in `renaissance/utils/write_*.py` — see `DATA.md` for dataset-specific instructions.

### Transforms (`renaissance/transforms/`)
`transform.py` and `randaug.py` handle image augmentation pipelines keyed by `train_transform_keys` / `val_transform_keys` config values (e.g., `"imagenet"`, `"clip"`).

## Data Preparation

Datasets must be converted to Arrow format before training. Conversion scripts are in `renaissance/utils/write_*.py`. Run the `make_arrow(root, arrows_root)` function for each dataset; place outputs under `data/arrow/`. See `DATA.md` for per-dataset download instructions.
