#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 18:42:56 2024

@author: claytonfields
"""

import random
import io
import pyarrow as pa
import os
import copy
import pytorch_lightning as pl
from sacred import Experiment
from PIL import Image
from tqdm import tqdm
import numpy as np
import skimage.io as skio
import matplotlib.pyplot as plt
from refer import REFER
import pandas as pd

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from transformers import ElectraTokenizer

from refcoco_utils import get_bounded_subimage
from refcoco_utils import _config
from refcoco_utils import _loss_names

from meter.transforms import keys_to_transforms
from meter.config import ex
from meter.modules import METERTransformerSS
from meter.datamodules.multitask_datamodule import MTDataModule
from meter.datasets.base_dataset import BaseDataset

# temporary variable switch between servers, fix before deployment
tensor_book = True
frege = False

if tensor_book:
    data_root =  "/home/claytonfields/nlp/code/meter/data/arrow"
    load_path = "/home/claytonfields/nlp/code/meter/result/mlm_itm_seed0_from_/meter_electra_small_deit_tiny_p16_is224_bs288_is1M/checkpoints/epoch=43-step=898039.ckpt"
    refer_root = "/home/claytonfields/nlp/code/data/coco"
    device = torch.device('cpu')
    num_gpus = 1
else:
    data_root =  "/data/clayton/meter/data/arrow"
    load_path = "/data/clayton/meter/result/meter_electra_small_deit_tiny_p16_is224_bs288_ts1M/checkpoints/epoch=43-step=898039.ckpt"
    refer_root = "/data/clayton/datasets/coco"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if frege:
        num_gpus = 2
    else:
        num_gpus = 1

_config = {  
    "exp_name":"finetune_ref",
    "seed" : 0,
    # "datasets" : ["coco", "vg", "sbu", "gcc"],
    # "datasets" : ["coco", "vg"],
    "datasets" : ["coco"],
    "loss_names" :{'itm': 1,
    'mlm': 1,
    'mpp': 0,
    'vqa': 0,
    'vcr': 0,
    'vcr_qar': 0,
    'nlvr2': 0,
    'irtr': 0,
    'contras': 0,
    'snli': 0,
    'ref': 0},
    "batch_size" : 128,  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.

    # Image setting
    "train_transform_keys" : ["imagenet"],
    "val_transform_keys" : ["imagenet"],
    "image_size" : 224,
    "patch_size" : 16,
    "draw_false_image" : 1,
    "image_only" : False,
    "resolution_before" : 224,

    # Text Setting
    "vqav2_label_size" : 3129,
    "max_text_len" : 40,
    "tokenizer" : "google/electra-small-discriminator",
    "vocab_size" : 30522,
    "whole_word_masking" : False, # note that whole_word_masking does not work for RoBERTa
    "mlm_prob" : 0.15,
    "draw_false_text" : 0,

    # Transformer Setting
    "num_top_layer" : 6,
    "input_image_embed_size" : 192,
    "input_text_embed_size" : 256,
    "vit" : "vit_deit_tiny_patch16_224",
    "hidden_size" : 256,
    "num_heads" : 4,
    "num_layers" : 6,
    "mlp_ratio" : 4,
    "drop_rate" : 0.1,

    # Optimizer Setting
    "optim_type" : "adamw",
    "learning_rate" : 1e-5,
    "weight_decay" : 0.01,
    "decay_power" : 1,
    "max_epoch" : 3,
    "max_steps" : 100000,
    "warmup_steps" : 10000,
    "end_lr" : 0,
    "lr_mult_head" : 5,  # multiply lr for downstream heads
    "lr_mult_cross_modal" : 5,  # multiply lr for the cross-modal module

    # Downstream Setting
    "get_recall_metric" : False,
    
    # Trainable parameter setting
    'freeze_image_encoder' : True,
    'freeze_text_encoder' : True,
    
    "model_type" : "METER",

    # PL Trainer Setting
    "resume_from" : None,
    "fast_dev_run" : False,
    "val_check_interval" : 1.0,
    "test_only" : False,

    "data_root" : data_root,
    "log_dir" : "result",
    "per_gpu_batchsize" : 128,  # you should define this manually with per_gpu_batch_size:#
    "num_gpus" : num_gpus,
    "num_nodes" : 1,
    "load_path" : load_path,
    "num_workers" : 12,
    "precision" : 32
}

pl.seed_everything(_config["seed"])

dm = MTDataModule(_config, dist=False)

model = METERTransformerSS(_config)
exp_name = f'{_config["exp_name"]}'

os.makedirs(_config["log_dir"], exist_ok=True)
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    save_top_k=1,
    verbose=True,
    monitor="val/the_metric",
    mode="max",
    save_last=True,
)
logger = pl.loggers.TensorBoardLogger(
    _config["log_dir"],
    name=f'{exp_name}_seed{_config["seed"]}_from_{_config["load_path"].split("/")[-1][:-5]}',
)

lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
callbacks = [checkpoint_callback, lr_callback]

num_gpus = (
    _config["num_gpus"]
    if isinstance(_config["num_gpus"], int)
    else len(_config["num_gpus"])
)

grad_steps = max(_config["batch_size"] // (
    _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
), 1)

max_steps = _config["max_steps"] if _config["max_steps"] is not None else None

trainer = pl.Trainer(
    gpus=0,
    num_nodes=_config["num_nodes"],
    precision=_config["precision"],
#     accelerator="cpu",
    benchmark=True,
    deterministic=True,
    max_epochs=_config["max_epoch"] if max_steps is None else 1000,
    max_steps=max_steps,
    callbacks=callbacks,
    logger=logger,
    #prepare_data_per_node=False,
    #replace_sampler_ddp=False,
    accumulate_grad_batches=grad_steps,
    log_every_n_steps=10,
    flush_logs_every_n_steps=10,
#     resume_from_checkpoint=_config["resume_from"],
    weights_summary="top",
    fast_dev_run=_config["fast_dev_run"],
    val_check_interval=_config["val_check_interval"],
)

if not _config["test_only"]:
    trainer.fit(model, datamodule=dm)
else:
    trainer.test(model, datamodule=dm)