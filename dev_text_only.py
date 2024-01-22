#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 16:03:26 2024

@author: claytonfields
"""

import random
import torch
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
from torch.utils.data import DataLoader

from torch.optim import AdamW

from transformers import ElectraTokenizer

from refcoco_utils import get_bounded_subimage
from refcoco_utils import _config
from refcoco_utils import _loss_names

from meter.transforms import keys_to_transforms
from meter.config import ex
from meter.modules import METERTransformerSS
from meter.datamodules.multitask_datamodule import MTDataModule
from meter.datasets.base_dataset import BaseDataset

from transformers import AutoTokenizer
from datasets import load_dataset
from torchtext.datasets import mrpc


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
    'ref': 0,
    'mrpc':1
    },
    "batch_size" : 10,  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.

    # Image setting
    "image_encoder" : "facebook/deit-tiny-patch16-224",
    "random_init_vision_encoder" : False,
    "image_encoder_hidden_size" : 192,
    "image_size" : 224,
    "patch_size" : 16,
    "draw_false_image" : 1,
    "image_only" : False,
    "resolution_before" : 224,
    "train_transform_keys" : ["imagenet"],
    "val_transform_keys" : ["imagenet"],

    # Text Setting
    "text_encoder" : "google/electra-small-discriminator",
    "random_init_text_encoder" : False,
    "text_encoder_hidden_size" : 256,
    "vocab_size" : 30522,
    "whole_word_masking" : False, # note that whole_word_masking does not work for RoBERTa
    "mlm_prob" : 0.15,
    "draw_false_text" : 0,
    "vqav2_label_size" : 3129,
    "max_text_len" : 40,

    # CrossLayer Setting
    "num_cross_layers" : 6,
    "cross_layer_hidden_size" : 256,
    "num_cross_layer_heads" : 4,
    "cross_layer_mlp_ratio" : 4,
    "cross_layer_drop_rate" : 0.1,
    
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

    # Encoder Settings
    "freeze_image_encoder" : False,
    "freeze_text_encoder" : False,
    

    # Downstream Setting
    "get_recall_metric" : False,
    
    'freeze' : True,
    
    "model_type" : "METER",

    # PL Trainer Setting
    "resume_from" : None,
    "fast_dev_run" : False,
    "val_check_interval" : 1.0,
    "test_only" : False,

    "data_root" : "/home/claytonfields/nlp/code/meter/data/arrow",
    "log_dir" : "result",
    "per_gpu_batchsize" : 3,  # you should define this manually with per_gpu_batch_size:#
    "num_gpus" : 1,
    "num_nodes" : 1,
    "load_path" : "/home/claytonfields/nlp/code/meter/result/mlm_itm_seed0_from_/meter_electra_small_deit_tiny_p16_is224_bs288_is1M/checkpoints/epoch=43-step=898039.ckpt",
    "num_workers" : 12,
    "precision" : 32
}

model = METERTransformerSS(_config)
# dm = MTDataModule(_config, dist=False)
# dm.prepare_data()
# dm.setup('train')

tokenizer = AutoTokenizer.from_pretrained(model.text_encoder.config._name_or_path)


class GlueDataset(torch.utils.data.Dataset):
    def __init__(self, task, split, tokenizer):

        self.task = task
        self.split = split
        self.tokenizer = tokenizer

        self.data_dict = load_dataset('glue', self.task, split=self.split).to_dict()
        self.sentence1 = self.data_dict['sentence1']
        self.sentence2 = self.data_dict['sentence2']
        self.label = self.data_dict['label']
        self.idx = self.data_dict['idx']

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):

        sent1 = self.sentence1[index]
        sent2 = self.sentence2[index]
        label = self.label[index]
        # idx = self.idx[index]

        ret = tokenizer(
            sent1, 
            sent2,
            max_length=60,
            padding='max_length',
            return_tensors='pt'
        )
        ret['label'] = label
        return ret


ds = GlueDataset('mrpc', 'train', tokenizer)
dl = DataLoader(ds, batch_size=10)

batch = next(iter(dl))






























