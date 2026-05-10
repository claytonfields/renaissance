# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## Running Experiments

All training and evaluation is driven through `run.py` using [Sacred](https://sacred.readthedocs.io/) named configs:

```bash
# Pretrain one-tower (any HF encoder)
python run.py with task_mlm_itm encoder=facebook/dino-vits16 max_steps=50000 \
  num_gpus=1 per_gpu_batchsize=32 batch_size=256 data_root=data/arrow/

# Pretrain two-tower
python run.py with task_mlm_itm image_encoder=facebook/deit-tiny-patch16-224 \
  text_encoder=google/electra-small-discriminator \
  cross_layer_hidden_size=256 num_cross_layers=6 \
  max_steps=50000 num_gpus=1 per_gpu_batchsize=32 batch_size=256 data_root=data/arrow/

# Fine-tune (replace task_mlm_itm with task_finetune_nlvr2 / task_finetune_vqa / task_finetune_snli)
python run.py with task_finetune_nlvr2 load_path=<CKPT> image_encoder=... text_encoder=... \
  cross_layer_hidden_size=256 num_cross_layers=6 image_size=288 \
  per_gpu_batchsize=32 num_gpus=1 num_nodes=1 data_root=data/arrow/

# Test only
python run.py with task_finetune_vqa load_path=<CKPT> test_only=True ...
```

Gradient accumulation is computed automatically: `grad_steps = batch_size / (per_gpu_batchsize × num_gpus × num_nodes)`.

Results and TensorBoard logs are written to `result/<exp_name>_seed<N>_is<img>_ps<patch>_bs<bs>_pgbs<pgbs>_ts<steps>/`.

## Configuration System

All config lives in `renaissance/config.py` (Sacred `@ex.config` + `@ex.named_config`). Any key in `config.py` can be overridden on the command line with `with key=value`. Named configs (functions decorated with `@ex.named_config`) bundle common settings:

```bash
python run.py with task_mlm_itm_deit_electra   # runs a predefined named config
```

Key config groups:
- **Model type**: `model_type = 'one-tower' | 'two-tower'`
- **One-tower**: `encoder`, `pooler_type` (`single` | `double`), `random_init_encoder`, `encoder_manual_configuration` + dimension overrides
- **Two-tower**: `image_encoder`, `text_encoder`, `random_init_vision_encoder`, `random_init_text_encoder`, `freeze_image_encoder`, `freeze_text_encoder`, cross-modal dims (`cross_layer_hidden_size`, `num_cross_layers`, `num_cross_layer_heads`, `cross_layer_mlp_ratio`, `cross_layer_drop_rate`)
- **Training**: `max_steps`, `max_epoch`, `batch_size`, `per_gpu_batchsize`, `learning_rate`, `warmup_steps`, `resume_from`
- **Tasks / losses**: `loss_names` dict — set any task to `1` to activate it (`mlm`, `itm`, `vqa`, `nlvr2`, `snli`, `irtr`, `ref`, `ref2`, `mrpc`, `rte`, `mnli`, `cifar10`, etc.)

## Architecture Overview

### Entry point
`run.py` → instantiates `MTDataModule` + `RenaissanceTransformer`, then hands off to a PyTorch Lightning `Trainer`.

### `RenaissanceTransformer` (`renaissance/modules/renaissance_module.py`)
The top-level `pl.LightningModule`. Delegates all encoding to one of two encoder classes and adds task-specific heads on top. Task routing happens in `forward()` via `self.current_tasks`, which is set each step by `renaissance_utils.set_task()`.

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
