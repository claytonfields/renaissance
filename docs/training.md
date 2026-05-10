# Training

## Basic usage

```bash
python run.py <config.yaml> [key=value overrides...]
```

Run `python run.py --help` for a full list of options.

---

## Gradient accumulation

Gradient accumulation is computed automatically from batch size parameters:

```
grad_steps = batch_size / (per_gpu_batchsize × num_gpus × num_nodes)
```

Example: `batch_size=256`, `per_gpu_batchsize=32`, `num_gpus=1` → `grad_steps=8`.

To achieve a larger effective batch size on fewer GPUs, increase `batch_size` or decrease `per_gpu_batchsize`.

---

## Distributed training

Renaissance uses HuggingFace Accelerate for distributed training. Launch with `torchrun` or `accelerate launch`:

```bash
# Single node, 4 GPUs
torchrun --nproc_per_node=4 run.py configs/pretrain_two_tower.yaml \
  training.num_gpus=4

# Multi-node (2 nodes, 8 GPUs each)
torchrun \
  --nnodes=2 \
  --nproc_per_node=8 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  run.py configs/pretrain_two_tower.yaml \
  training.num_gpus=8 \
  training.num_nodes=2
```

For multi-node jobs, set the environment variables `MASTER_ADDR`, `MASTER_PORT`, and `NODE_RANK` before launching.

---

## Mixed precision

Control via `training.precision`:

| Value | Effect |
|-------|--------|
| `32` | Full precision (default) |
| `16` | FP16 — faster on Volta/Ampere, may require loss scaling |
| `"bf16"` | BFloat16 — more stable than FP16, requires Ampere+ |

```bash
python run.py configs/pretrain_two_tower.yaml training.precision=16
```

---

## Resuming from a checkpoint

Each call to `save_checkpoint` writes two artifacts:

```
result/my_experiment/
├── config.json          # model architecture config
├── model.safetensors    # model weights
└── training_state/      # optimizer + scheduler + RNG state
```

To resume training from where it left off, set `experiment.resume_from` to the `training_state/` directory:

```bash
python run.py configs/pretrain_two_tower.yaml \
  experiment.resume_from=result/pretrain_two_tower_.../training_state/
```

To load weights for fine-tuning (not resume), use `experiment.load_path` instead:

```bash
python run.py configs/finetune_nlvr2.yaml \
  experiment.load_path=result/pretrain_two_tower_.../
```

`load_path` accepts either a local directory (containing `model.safetensors` + `config.json`) or a HuggingFace Hub repo id.

---

## LR schedule

Two schedule types are supported:

**Polynomial decay** (`decay_power` = integer):
- LR rises linearly from 0 to `learning_rate` over `warmup_steps`.
- LR decays polynomially to `end_lr` over `max_steps`.

**Cosine decay** (`decay_power = "cosine"`):
- Same warmup, then cosine decay to 0.
- Recommended for fine-tuning tasks.

```bash
# Cosine decay with 10% warmup
python run.py configs/finetune_nlvr2.yaml \
  training.decay_power=cosine \
  training.warmup_steps=0.1
```

---

## Learning rate groups

Task-specific heads and the cross-modal encoder use higher LR multipliers to allow faster adaptation when fine-tuning from a pretrained backbone:

| Parameter group | LR |
|-----------------|-----|
| Backbone encoders | `learning_rate` |
| Cross-modal fusion | `learning_rate × lr_mult_cross_modal` (default 5×) |
| Task heads | `learning_rate × lr_mult_head` (default 5×) |

Adjust via `training.lr_mult_head` and `training.lr_mult_cross_modal`.

---

## TensorBoard logs

Training and validation metrics are written to TensorBoard under `log_dir`:

```bash
tensorboard --logdir result/
```

Step-level metrics (loss, accuracy) are flushed each step. Epoch-level metrics are written at the end of each epoch.

---

## Result directory naming

Results are saved to:

```
result/<exp_name>_seed<seed>_is<image_size>_ps<patch_size>_bs<batch_size>_pgbs<per_gpu_batchsize>_ts<max_steps>/
```

Example:
```
result/pretrain_two_tower_seed0_is224_ps16_bs256_pgbs32_ts100000/
```

---

## Freezing encoders

To freeze the image or text encoder (e.g., train only the cross-modal fusion):

```bash
python run.py configs/pretrain_two_tower.yaml \
  model.freeze_image_encoder=true \
  model.freeze_text_encoder=true
```

---

## Tips

- **OOM:** Reduce `per_gpu_batchsize` and increase `batch_size` proportionally to maintain effective batch size.
- **Speed:** Enable `training.precision=16` or `"bf16"` on supported hardware.
- **Stability:** If fp16 training diverges, switch to `"bf16"` or lower `learning_rate`.
- **Fine-tuning:** Use a higher image resolution (`model.image_size=288`) and cosine schedule. Renaissance interpolates position embeddings automatically when `image_size != original_image_size`.
