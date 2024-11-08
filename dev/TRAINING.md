## Pre-training

```bash
python run.py with task_mlm_twotower image_encoder=<IMAGE_ENCODER> text_encoder=<TEXT_ENCODER> cross_layer_hidden_size=<CROSS_LAYER_HIDDEN_SIZE> num_cross_layers=<NUM_CROSS_LAYER> max_steps=<TRAINING_STEPS> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> per_gpu_batchsize=<BS_FITS_YOUR_GPU> batch_size=<BATCH_SIZE>

  data_root=<ARROW_ROOT>
```

Here is an example:
```bash
python3 run.py with task_mlm_twotower image_encoder=facebook/deit-tiny-patch16-224 text_encoder=google/electra-small-discriminator cross_layer_hidden_size=256 num_cross_layers=6 max_steps=50000 num_gpus=1 num_nodes=1 per_gpu_batchsize=32 batch_size=256 data_root=data/arrow/
``` 

## Fine-tuning on Downstream Tasks

### VQAv2

```bash
export MASTER_ADDR=$DIST_0_IP
export MASTER_PORT=$DIST_0_PORT
export NODE_RANK=$DIST_RANK
python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> task_finetune_vqa_clip_bert per_gpu_batchsize=<BS_FITS_YOUR_GPU> load_path=<PRETRAINED_MODEL> <IMAGE_ENCODER> <TEXT_ENCODER> image_size=<IMAGE_SIZE> <IMAGE_AUGMENTATION>
```

Here is an example:
```bash
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1 task_finetune_vqa_clip_bert per_gpu_batchsize=32 load_path=meter_pretrain.ckpt clip16 text_roberta image_size=576 clip_randaug 
``` 

### Flickr30k IR/TR

```bash
export MASTER_ADDR=$DIST_0_IP
export MASTER_PORT=$DIST_0_PORT
export NODE_RANK=$DIST_RANK
python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> task_finetune_irtr_f30k_clip_bert get_recall_metric=False per_gpu_batchsize=<BS_FITS_YOUR_GPU> load_path=<PRETRAINED_MODEL> <IMAGE_ENCODER> <TEXT_ENCODER> image_size=<IMAGE_SIZE> <IMAGE_AUGMENTATION>
```

Here is an example:
```bash
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1 task_finetune_irtr_f30k_clip_bert get_recall_metric=False per_gpu_batchsize=32 load_path=meter_pretrain.ckpt clip16 text_roberta image_size=384 clip_randaug 
``` 

### COCO IR/TR

```bash
export MASTER_ADDR=$DIST_0_IP
export MASTER_PORT=$DIST_0_PORT
export NODE_RANK=$DIST_RANK
python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> task_finetune_irtr_coco_clip_bert get_recall_metric=False per_gpu_batchsize=<BS_FITS_YOUR_GPU> load_path=<PRETRAINED_MODEL> <IMAGE_ENCODER> <TEXT_ENCODER> image_size=<IMAGE_SIZE> <IMAGE_AUGMENTATION>
```

Here is an example:
```bash
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1 task_finetune_irtr_coco_clip_bert get_recall_metric=False per_gpu_batchsize=32 load_path=meter_pretrain.ckpt clip16 text_roberta image_size=384 clip_randaug 
``` 

### NLVR2

```bash
export MASTER_ADDR=$DIST_0_IP
export MASTER_PORT=$DIST_0_PORT
export NODE_RANK=$DIST_RANK
python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES>  task_finetune_nlvr2_clip_bert per_gpu_batchsize=<BS_FITS_YOUR_GPU> load_path=<PRETRAINED_MODEL> <IMAGE_ENCODER> <TEXT_ENCODER> image_size=<IMAGE_SIZE> <IMAGE_AUGMENTATION>
```

Here is an example:
```bash
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1  task_finetune_nlvr2_clip_bert per_gpu_batchsize=32 load_path=meter_pretrain.ckpt clip16 text_roberta image_size=288 clip_randaug 
``` 

### SNLI-VE

```bash
export MASTER_ADDR=$DIST_0_IP
export MASTER_PORT=$DIST_0_PORT
export NODE_RANK=$DIST_RANK
python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> task_finetune_snli_clip_bert per_gpu_batchsize=<BS_FITS_YOUR_GPU> load_path=<PRETRAINED_MODEL> <IMAGE_ENCODER> <TEXT_ENCODER> image_size=<IMAGE_SIZE> <IMAGE_AUGMENTATION>
```

Here is an example:
```bash
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1 task_finetune_snli_clip_bert per_gpu_batchsize=8 load_path=meter_pretrain.ckpt clip16 text_roberta image_size=384 clip_randaug 
``` 

## Evaluation on Downstream Tasks

### VQAv2

```bash
export MASTER_ADDR=$DIST_0_IP
export MASTER_PORT=$DIST_0_PORT
export NODE_RANK=$DIST_RANK
python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> task_finetune_vqa_clip_bert per_gpu_batchsize=<BS_FITS_YOUR_GPU> load_path=<PRETRAINED_MODEL> <IMAGE_ENCODER> <TEXT_ENCODER> image_size=<IMAGE_SIZE> test_only=True
```

Here is an example:
```bash
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1 task_finetune_vqa_clip_bert per_gpu_batchsize=32 load_path=meter_vqa.ckpt clip16 text_roberta image_size=576 test_only=True
``` 

Then, submit the json file in the `result` directory to [eval.ai](https://eval.ai/web/challenges/challenge-page/830/overview) evaluation server to get the test-dev and/or test-std scores.


### Flickr30k IR/TR

```bash
export MASTER_ADDR=$DIST_0_IP
export MASTER_PORT=$DIST_0_PORT
export NODE_RANK=$DIST_RANK
python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> task_finetune_irtr_f30k_clip_bert get_recall_metric=True per_gpu_batchsize=<BS_FITS_YOUR_GPU> load_path=<PRETRAINED_MODEL> <IMAGE_ENCODER> <TEXT_ENCODER> image_size=<IMAGE_SIZE> test_only=True
```

Here is an example:
```bash
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1 task_finetune_irtr_f30k_clip_bert get_recall_metric=True per_gpu_batchsize=32 load_path=meter_f30k.ckpt clip16 text_roberta image_size=384 test_only=True
``` 

The returned values are IR R@1, R@5, R@10 and TR R@1, R@5, R@10.

### COCO IR/TR

```bash
export MASTER_ADDR=$DIST_0_IP
export MASTER_PORT=$DIST_0_PORT
export NODE_RANK=$DIST_RANK
python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> task_finetune_irtr_coco_clip_bert get_recall_metric=True per_gpu_batchsize=<BS_FITS_YOUR_GPU> load_path=<PRETRAINED_MODEL> <IMAGE_ENCODER> <TEXT_ENCODER> image_size=<IMAGE_SIZE> test_only=True
```

Here is an example:
```bash
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1 task_finetune_irtr_coco_clip_bert get_recall_metric=True per_gpu_batchsize=32 load_path=meter_coco.ckpt clip16 text_roberta image_size=384 test_only=True
``` 

The returned values are IR R@1, R@5, R@10 and TR R@1, R@5, R@10.

### NLVR2

```bash
export MASTER_ADDR=$DIST_0_IP
export MASTER_PORT=$DIST_0_PORT
export NODE_RANK=$DIST_RANK
python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES>  task_finetune_nlvr2_clip_bert per_gpu_batchsize=<BS_FITS_YOUR_GPU> load_path=<PRETRAINED_MODEL> <IMAGE_ENCODER> <TEXT_ENCODER> image_size=<IMAGE_SIZE> test_only=True
```

Here is an example:
```bash
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1  task_finetune_nlvr2_clip_bert per_gpu_batchsize=32 load_path=meter_nlvr2.ckpt clip16 text_roberta image_size=288 test_only=True
``` 

### SNLI-VE

```bash
export MASTER_ADDR=$DIST_0_IP
export MASTER_PORT=$DIST_0_PORT
export NODE_RANK=$DIST_RANK
python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> task_finetune_snli_clip_bert per_gpu_batchsize=<BS_FITS_YOUR_GPU> load_path=<PRETRAINED_MODEL> <IMAGE_ENCODER> <TEXT_ENCODER> image_size=<IMAGE_SIZE> test_only=True
```

Here is an example:
```bash
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1 task_finetune_snli_clip_bert per_gpu_batchsize=8 load_path=meter_snli.ckpt clip16 text_roberta image_size=384 test_only=True