# Configuration

Renaissance uses typed dataclasses (`renaissance/config_schema.py`) and YAML files under `configs/`. The schema has five groups — `experiment`, `model`, `task`, `data`, `training` — which are flattened into a plain dict before being passed to `RenaissanceTransformer`.

## Loading a config

```bash
python run.py configs/pretrain_two_tower.yaml key=value key2=value2
```

CLI overrides use dotted paths matching the group structure:

```bash
python run.py configs/pretrain_two_tower.yaml \
  model.image_encoder=openai/clip-vit-base-patch32 \
  training.learning_rate=2e-5 \
  training.max_steps=50000
```

---

## Config groups

### `experiment`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `exp_name` | str | `"renaissance"` | Experiment name, used in the result directory. |
| `seed` | int | `0` | Random seed. |
| `load_path` | str | `""` | Path to a checkpoint to fine-tune or test. |
| `resume_from` | str \| None | `None` | Path to a `training_state/` directory to resume training. |
| `test_only` | bool | `false` | If `true` + `load_path` set, skip training and run evaluation only. |
| `log_dir` | str | `"result"` | Root directory for TensorBoard logs and checkpoints. |

---

### `model`

#### Common

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_type` | str | `"two-tower"` | Architecture: `"one-tower"` or `"two-tower"`. |

#### One-tower fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `encoder` | str | `"google/electra-small-discriminator"` | HF model id for the shared encoder. |
| `pooler_type` | str | `"double"` | CLS pooling: `"single"` (one CLS) or `"double"` (text + image CLS concatenated). |
| `random_init_encoder` | bool | `false` | Randomly initialise encoder weights (skip downloading pretrained). |
| `encoder_manual_configuration` | bool | `false` | Use manual dimension overrides instead of reading from HF config. |
| `hidden_size` | int | `192` | Encoder hidden size (only used when `encoder_manual_configuration=true`). |
| `num_heads` | int | `4` | Number of attention heads (manual config only). |
| `num_layers` | int | `12` | Number of transformer layers (manual config only). |
| `mlp_ratio` | int | `4` | FFN hidden size multiplier (manual config only). |
| `drop_rate` | float | `0.1` | Dropout rate (manual config only). |
| `embedding_size` | int | `96` | Patch/token embedding size (manual config only). |

#### Two-tower image encoder fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `image_encoder` | str | `"facebook/deit-tiny-patch16-224"` | HF model id for the vision encoder. |
| `random_init_vision_encoder` | bool | `false` | Randomly initialise vision encoder. |
| `image_encoder_manual_configuration` | bool | `false` | Use manual dimension overrides for vision encoder. |
| `image_encoder_hidden_size` | int | `192` | Vision encoder hidden size (manual config only). |
| `image_encoder_num_heads` | int | `4` | (manual config only) |
| `image_encoder_num_layers` | int | `12` | (manual config only) |
| `image_encoder_mlp_ratio` | int | `4` | (manual config only) |
| `image_encoder_drop_rate` | float | `0.1` | (manual config only) |
| `image_encoder_embedding_size` | int | `128` | (manual config only) |
| `image_size` | int | `224` | Input image resolution. Use `288` for fine-tuning tasks. |
| `original_image_size` | int | `224` | Image size the model was pretrained with (used for position embedding interpolation). |
| `patch_size` | int | `16` | Patch size in pixels. |
| `image_only` | bool | `false` | Image-only classification mode (e.g. CIFAR-10). |

#### Two-tower text encoder fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text_encoder` | str | `"google/electra-small-discriminator"` | HF model id for the text encoder. |
| `random_init_text_encoder` | bool | `false` | Randomly initialise text encoder. |
| `text_encoder_manual_configuration` | bool | `false` | Use manual dimension overrides for text encoder. |
| `text_encoder_hidden_size` | int | `192` | (manual config only) |
| `text_encoder_num_heads` | int | `4` | (manual config only) |
| `text_encoder_num_layers` | int | `12` | (manual config only) |
| `text_encoder_mlp_ratio` | int | `4` | (manual config only) |
| `text_encoder_drop_rate` | float | `0.1` | (manual config only) |
| `text_encoder_embedding_size` | int | `64` | (manual config only) |
| `max_text_len` | int | `40` | Maximum token sequence length. |
| `vocab_size` | int | `30522` | Vocabulary size (must match tokeniser). |

#### Cross-modal fusion fields (two-tower only)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `cross_layer_hidden_size` | int | `256` | Hidden size of the cross-modal fusion encoder. |
| `num_cross_layers` | int | `6` | Number of cross-attention layers. |
| `num_cross_layer_heads` | int | `4` | Number of attention heads in cross-modal layers. |
| `cross_layer_mlp_ratio` | int | `4` | FFN hidden size multiplier in cross-modal layers. |
| `cross_layer_drop_rate` | float | `0.1` | Dropout rate in cross-modal layers. |

#### Freeze flags

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `freeze_image_encoder` | bool | `false` | Freeze image encoder weights during training. |
| `freeze_text_encoder` | bool | `false` | Freeze text encoder weights during training. |
| `freeze_cross_modal_layers` | bool | `false` | Freeze cross-modal fusion encoder weights. |

---

### `task`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `loss_names` | dict[str, int] | all zeros except `itm: 1, mlm: 1` | Set any task to `1` to activate it. |
| `mlm_prob` | float | `0.15` | Masking probability for MLM. |
| `whole_word_masking` | bool | `false` | Mask whole words rather than individual tokens. |
| `draw_false_image` | int | `1` | Number of negative images per sample (for ITM). |
| `draw_false_text` | int | `0` | Number of negative texts per sample (for IRTR). |
| `get_recall_metric` | bool | `false` | Compute IRTR recall@k on the validation set each epoch. |
| `vqav2_label_size` | int | `3129` | Number of VQAv2 answer classes. |
| `max_bb` | int | `20` | Maximum number of bounding-box regions (ref resolution). |
| `ref_res_head_layers` | int | `2` | Depth of the reference resolution classification head. |

Available task keys: `itm`, `mlm`, `vqa`, `nlvr2`, `snli`, `irtr`, `ref`, `ref2`, `mrpc`, `rte`, `wnli`, `sst2`, `qqp`, `qnli`, `mnli`, `cola`, `cifar10`.

---

### `data`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `datasets` | list[str] | `["coco", "vg"]` | Dataset keys to use. See `docs/data-preparation.md`. |
| `data_root` | str | `"data/arrow/"` | Root directory containing converted Arrow files. |
| `train_transform_keys` | list[str] | `["imagenet"]` | Image augmentation pipeline for training. Options: `"imagenet"`, `"clip"`. |
| `val_transform_keys` | list[str] | `["imagenet"]` | Image transform for validation. |
| `batch_size` | int | `256` | Global batch size (across all GPUs and nodes). |
| `per_gpu_batchsize` | int | `32` | Per-GPU batch size. Gradient accumulation = `batch_size / (per_gpu_batchsize × num_gpus × num_nodes)`. |
| `eval_batch_size` | int | `32` | Batch size for validation. |
| `num_workers` | int | `12` | DataLoader worker processes per GPU. |

---

### `training`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `optim_type` | str | `"adamw"` | Optimiser: `"adamw"`, `"adam"`, or `"sgd"`. |
| `learning_rate` | float | `1e-5` | Peak learning rate. |
| `weight_decay` | float | `0.01` | Weight decay (AdamW). |
| `decay_power` | int \| str | `1` | LR schedule: integer for polynomial decay, `"cosine"` for cosine. |
| `max_epoch` | int | `100` | Maximum number of training epochs. |
| `max_steps` | int | `100000` | Maximum number of training steps. Training stops at whichever limit is reached first. |
| `warmup_steps` | int \| float | `10000` | Warmup steps. If a float `< 1.0`, treated as a fraction of `max_steps`. |
| `end_lr` | float | `0.0` | Final learning rate after decay. |
| `lr_mult_head` | float | `5.0` | LR multiplier for task-specific heads. |
| `lr_mult_cross_modal` | float | `5.0` | LR multiplier for cross-modal fusion encoder. |
| `precision` | int \| str | `32` | Training precision: `32` (full), `16` (fp16), or `"bf16"`. |
| `num_gpus` | int \| list | `1` | Number of GPUs per node. |
| `num_nodes` | int | `1` | Number of training nodes. |
| `val_check_interval` | float | `1.0` | Run validation every N epochs (float = fraction of epoch). |
| `fast_dev_run` | bool | `false` | Reserved. |

---

## YAML examples

### Minimal pretrain (one-tower)

```yaml
experiment:
  exp_name: one_tower_dino

model:
  model_type: one-tower
  encoder: facebook/dino-vits16

task:
  loss_names:
    itm: 1
    mlm: 1

data:
  datasets: [coco, vg]
  data_root: data/arrow/
  batch_size: 256
  per_gpu_batchsize: 32

training:
  max_steps: 50000
  learning_rate: 1.0e-5
  num_gpus: 1
```

### Fine-tune VQAv2

```yaml
experiment:
  exp_name: vqa_finetune
  load_path: result/pretrain_two_tower_.../model.safetensors

model:
  model_type: two-tower
  image_size: 288
  original_image_size: 224

task:
  loss_names:
    vqa: 1
  vqav2_label_size: 3129

data:
  datasets: [vqa]
  batch_size: 512
  per_gpu_batchsize: 32

training:
  max_steps: 100000
  decay_power: cosine
  warmup_steps: 0.1
```
