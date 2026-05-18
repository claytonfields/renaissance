# Renaissance

A multimodal vision-language modeling platform built on HuggingFace Transformers and Accelerate. Plug any HF encoder into a one-tower or two-tower architecture and pretrain / fine-tune on standard VLP benchmarks with a single command.

## Quick install

```bash
pip install -r requirements.txt
pip install -e .
```

## Five-command quickstart

```bash
# 1. Prepare data (see docs/data-preparation.md)
python -c "from renaissance.utils.legacy.write_coco_karpathy import make_arrow; make_arrow('data/coco/', 'data/arrow/')"

# 2. Pretrain two-tower (DeiT-Tiny + ELECTRA-Small)
python run.py configs/pretrain_two_tower.yaml \
  data.data_root=data/arrow/ training.num_gpus=1

# 3. Fine-tune on NLVR2
python run.py configs/finetune_nlvr2.yaml \
  experiment.load_path=result/pretrain_two_tower_seed0_is224_ps16_bs256_pgbs32_ts100000/model.safetensors \
  data.data_root=data/arrow/

# 4. Evaluate
python -m renaissance.eval \
  --checkpoint result/finetune_nlvr2_seed0_is288_ps16_bs128_pgbs32_ts25000/ \
  --task nlvr2 --split val --data_root data/arrow/

# 5. Push to Hub
python -c "
from renaissance.modeling import RenaissanceModel
from renaissance.hub import push_to_hub
model = RenaissanceModel.from_pretrained('result/finetune_nlvr2_...')
push_to_hub(model, 'myuser/renaissance-nlvr2')
"
```

## Architecture

Renaissance supports two encoder configurations:

**One-tower** — a single HF transformer backbone shared for both text and image. Text uses BERT-style word-piece embeddings; images use ViT patch embeddings. Both streams are concatenated and fed through the shared encoder.

**Two-tower** — separate HF text and vision encoders whose outputs are fused by a learned cross-modal encoder (`LxmertCrossModalEncoder`). The fusion encoder is always trained from scratch; the backbone encoders can be frozen or fine-tuned.

## Documentation

| Document | Contents |
|----------|----------|
| [docs/data-preparation.md](docs/data-preparation.md) | Dataset downloads, Arrow conversion, directory layout |
| [docs/configuration.md](docs/configuration.md) | All config fields, defaults, YAML structure, CLI overrides |
| [docs/training.md](docs/training.md) | Distributed training, gradient accumulation, mixed precision, resuming |
| [docs/benchmarks.md](docs/benchmarks.md) | Expected metric values on standard val splits |

## Running experiments

All training is driven by `run.py` with a YAML config file and optional CLI dot-path overrides:

```bash
# Pretrain one-tower
python run.py configs/pretrain_one_tower.yaml \
  model.encoder=facebook/dino-vits16 \
  training.max_steps=50000

# Pretrain two-tower
python run.py configs/pretrain_two_tower.yaml \
  data.data_root=data/arrow/ \
  training.num_gpus=4

# Fine-tune on VQAv2
python run.py configs/finetune_vqa.yaml \
  experiment.load_path=<CHECKPOINT> \
  model.image_size=288

# Test only
python run.py configs/finetune_vqa.yaml \
  experiment.load_path=<CHECKPOINT> \
  experiment.test_only=true
```

Gradient accumulation is computed automatically:
`grad_steps = batch_size / (per_gpu_batchsize × num_gpus × num_nodes)`

Results and TensorBoard logs are written to:
`result/<exp_name>_seed<N>_is<img>_ps<patch>_bs<bs>_pgbs<pgbs>_ts<steps>/`

## Checkpoints

Checkpoints are stored in safetensors format with a companion `config.json`:

```
result/
└── my_experiment/
    ├── config.json          # model hyperparameters (RenaissanceHubConfig)
    ├── model.safetensors    # model weights
    └── training_state/      # optimizer + scheduler + RNG (for resuming)
```

Load a checkpoint programmatically:

```python
from renaissance.modeling import RenaissanceModel
model = RenaissanceModel.from_pretrained("result/my_experiment/")
```

## Evaluation

```bash
python -m renaissance.eval \
  --checkpoint result/my_experiment/ \
  --task snli \
  --split val \
  --data_root data/arrow/ \
  --output results.json
```

Supported tasks: `mlm`, `itm`, `vqa`, `nlvr2`, `snli`, `ref`, `ref2`, `mrpc`.

## Examples

See [`examples/pretrain_two_tower.ipynb`](examples/pretrain_two_tower.ipynb) for an end-to-end walkthrough using synthetic data on a single CPU/GPU.

## Acknowledgements

Built on top of [ViLT](https://github.com/dandelin/ViLT) and [METER](https://github.com/zdou0830/METER) (Apache 2.0). Encoder architectures from [HuggingFace Transformers](https://github.com/huggingface/transformers). Cross-modal fusion layer adapted from [LXMERT](https://github.com/airsplay/lxmert).
