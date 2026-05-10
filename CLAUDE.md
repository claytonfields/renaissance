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

Config lives in typed dataclasses (`renaissance/config_schema.py`) and YAML files under `configs/`. The schema has five groups — `experiment`, `model`, `task`, `data`, `training` — which are flattened into a plain dict before being passed to `RenaissanceTransformer`.

Key config groups:
- **Model type**: `model.model_type = 'one-tower' | 'two-tower'`
- **One-tower**: `model.encoder`, `model.pooler_type` (`single` | `double`), `model.random_init_encoder`, dimension overrides
- **Two-tower**: `model.image_encoder`, `model.text_encoder`, `model.freeze_image_encoder`, `model.freeze_text_encoder`, cross-modal dims (`model.cross_layer_hidden_size`, `model.num_cross_layers`, `model.num_cross_layer_heads`, `model.cross_layer_mlp_ratio`, `model.cross_layer_drop_rate`)
- **Training**: `training.max_steps`, `training.max_epoch`, `data.batch_size`, `data.per_gpu_batchsize`, `training.learning_rate`, `training.warmup_steps`
- **Tasks / losses**: `task.loss_names` dict — set any task to `1` to activate it (`mlm`, `itm`, `vqa`, `nlvr2`, `snli`, `irtr`, `ref`, `ref2`, `mrpc`, `rte`, `mnli`, `cifar10`, etc.)

The legacy Sacred-based config is preserved at `renaissance/config_legacy.py`.

## Architecture Overview

### Entry point
`run.py` → loads YAML + CLI overrides via omegaconf → instantiates `MTDataModule` + `RenaissanceTransformer` → hands off to `RenaissanceTrainer` (Accelerate-based).

### `RenaissanceTransformer` (`renaissance/modules/renaissance_module.py`)
Plain `nn.Module`. Delegates all encoding to one of two encoder classes and adds task-specific heads on top. Task routing happens in `forward()` via `self.current_tasks`, which is set each step by `renaissance_utils.set_task()`. Step-level metrics are buffered in `self._log_buffer`; the trainer flushes them to TensorBoard.

### Encoder classes (`renaissance/modules/`)
- **`OneTowerEncoder`** — shares a single HF transformer backbone for both modalities. Text uses custom `ElectraEmbeddings`; images use `ViTEmbeddings` (patch projection). Both embedding streams are concatenated and fed through the shared encoder. Supports `single` and `double` CLS pooling.
- **`TwoTowerEncoder`** — separate HF text (`text_transformer`) and vision (`image_encoder`) encoders, each downloaded via `AutoModel`. Their outputs are linearly projected to a shared `cross_layer_hidden_size`, then fused by `LxmertCrossModalEncoder` (parallel cross-attention layers). The fused CLS features from both streams are concatenated for downstream heads. Handles convolutional vision models via `resize_convolutional_output()`.
- **`LxmertCrossModalEncoder`** (`fusion_encoder.py`) — the cross-modal fusion module, always trained from scratch. Uses stacked `LxmertXLayer` cross-attention, with separate text and image poolers.
- **`nox_two_tower_encoder.py`** — new/in-progress encoder variant (untracked file on current branch).

### Heads (`renaissance/modules/heads.py`)
`MLMHead`, `ITMHead`, `MultiModalClassificationHead` (VQA, SNLI, ref), `NLVR2ClassificationHead`, `UniModalClassificationHead` (GLUE text tasks, CIFAR-10 image-only).

### Objectives (`renaissance/modules/objectives.py`)
One `compute_<task>()` function per task. Each reads from the batch dict and calls `self.infer()` or `self.infer_text_only()`.

### Data pipeline
`renaissance/datamodules/multitask_datamodule.py` (`MTDataModule`) coordinates multiple `DataModule` instances. Each dataset has a corresponding `*_datamodule.py` and `*_dataset.py`. Datasets are pre-serialized to [Apache Arrow](https://arrow.apache.org/) format using the scripts in `renaissance/utils/write_*.py` — see `DATA.md` for dataset-specific instructions.

### Transforms (`renaissance/transforms/`)
`transform.py` and `randaug.py` handle image augmentation pipelines keyed by `train_transform_keys` / `val_transform_keys` config values (e.g., `"imagenet"`, `"clip"`).

## Data Preparation

Datasets must be converted to Arrow format before training. Conversion scripts are in `renaissance/utils/write_*.py`. Run the `make_arrow(root, arrows_root)` function for each dataset; place outputs under `data/arrow/`. See `DATA.md` for per-dataset download instructions.
