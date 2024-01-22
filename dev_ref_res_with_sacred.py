#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 14:16:16 2024

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
# import matplotlib.pyplot as plt
from refer import REFER
import pandas as pd

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from transformers import ElectraTokenizer

from refcoco_utils import get_bounded_subimage
# from refcoco_utils import _config
# from refcoco_utils import _loss_names

from meter.transforms import keys_to_transforms
from meter.config import ex
from meter.modules import METERTransformerSS
from meter.datamodules.multitask_datamodule import MTDataModule
from meter.datasets.base_dataset import BaseDataset

# temporary variable switch between servers, fix before deployment
tensor_book = True
frege = False

if tensor_book:
    data_root =  "/home/claytonfields/nlp/code/vilt/data/arrow"
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
    "loss_names" :{'itm': 0,
    'mlm': 0,
    'mpp': 0,
    'vqa': 0,
    'vcr': 0,
    'vcr_qar': 0,
    'nlvr2': 0,
    'irtr': 0,
    'contras': 0,
    'snli': 0,
    'ref': 1},
    "batch_size" : 10,  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.

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
    
    
    "model_type" : "METER",

    # PL Trainer Setting
    "resume_from" : None,
    "fast_dev_run" : False,
    "val_check_interval" : 1.0,
    "test_only" : False,

    "data_root" : data_root,
    "log_dir" : "result",
    "per_gpu_batchsize" : 3,  # you should define this manually with per_gpu_batch_size:#
    "num_gpus" : num_gpus,
    "num_nodes" : 1,
    "load_path" : load_path,
    "num_workers" : 12,
    "precision" : 16
}