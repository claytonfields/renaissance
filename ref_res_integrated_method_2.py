#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 16:10:31 2023

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
    "max_epoch" : 100,
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
    "per_gpu_batchsize" : 5,  # you should define this manually with per_gpu_batch_size:#
    "num_gpus" : num_gpus,
    "num_nodes" : 1,
    "load_path" : load_path,
    "num_workers" : 12,
    "precision" : 16
}

dataset = 'refcoco' 
splitBy = 'unc'
refer = REFER(refer_root, dataset, splitBy)

class RefcocoDataset(torch.utils.data.Dataset):

    def __init__(self, refer, tokenizer, device, errors, split='', max_bb = 42):
        self.tokenizer = tokenizer
        self.refer = refer
        self.max_bb = max_bb
        self.device = device
        self.errors = errors
        self.split = split
        self.sent_ids = self.get_sent_ids()[:30]
        self.duds = []

    def __len__(self):
        return len(self.sent_ids)
    
    def get_sent_ids(self):
        sent_ids = []
        for ref_id in self.refer.getRefIds(split=self.split):
            
            ref = self.refer.Refs[ref_id]
            img_id = ref['image_id']
            objs = refer.imgToAnns[img_id]
            if len(objs) <= self.max_bb:
                for sent_id in ref['sent_ids']:
                    if not sent_id in self.errors:
                        sent_ids.append(sent_id)
        return sent_ids
    
    def __getitem__(self, index):
        max_bb = self.max_bb
        
        sent_id = self.sent_ids[index]
        ref = self.refer.sentToRef[sent_id]
        sent = self.refer.Sents[sent_id]
        
        img_id = ref['image_id']
        ann_id = ref['ann_id']
        objs = refer.imgToAnns[img_id]
        obj_ids = [obj['id'] for obj in objs]
        obj_pad = [0 for _ in range(max_bb-len(obj_ids))]
        obj_ids_total = obj_ids+obj_pad

        sub_images = []
        for obj in objs:
            x_a = get_bounded_subimage(refer, img_id, obj['id'], xs=224,ys=224, show=False)
            if x_a is not None:
                sub_images.append(x_a)
        
        num_sub_images = len(sub_images)
        num_pad = max_bb - num_sub_images 
        
        pad_image = torch.zeros(1,3,224,224)
        for _ in range(max_bb - num_sub_images):
            sub_images.append(pad_image)
        
        # text ids
        ids = self.tokenizer.encode(
            sent['sent'],
            padding="max_length",
            truncation=True,
            max_length=40,
            return_special_tokens_mask=True,
        )
        repeat_ids = torch.tensor(ids).repeat(num_sub_images,1)
        pad_ids =  torch.zeros(num_pad,40)
        text_ids = torch.cat((repeat_ids, pad_ids)).to(torch.long)
        # text masks
        num_tokens = torch.where(text_ids[0] > 0)[0].size(dim=0)
        masks = torch.cat((torch.ones(num_tokens), torch.zeros(40-num_tokens))).to(torch.long)
        repeat_masks = masks.repeat(num_sub_images,1)
        pad_masks = torch.zeros(num_pad, 40)
        text_masks = torch.cat((repeat_masks, pad_masks)).to(torch.long)
        # text_labels
        labels = torch.full((40,),-100)
        repeat_labels = labels.repeat(num_sub_images, 1)
        pad_labels = torch.zeros(num_pad, 40)
        text_labels = torch.cat((repeat_labels, pad_labels)).to(torch.long)
        
        target = torch.tensor([obj_ids.index(ann_id)])

        return_dict = {
            'ann_id' : ann_id,
            'image' : [torch.cat(sub_images)],#.to(self.device)],
            'obj_ids' : torch.tensor(obj_ids_total),#.to(self.device),
            'target' : target,#.to(self.device),
            'text' : sent['sent'],
            'text_ids' : text_ids,#.to(self.device),
            'text_labels' : text_labels,#.to(self.device),
            'text_masks' : text_masks,#.to(self.device)
        }
        
        return return_dict


def collate(batch):
    targets = []
    for b in batch:
        targets.append(b['target'])
    targets = torch.tensor(targets)
    return (batch, targets)

class RefcocoDataModule(LightningDataModule):
    def __init__(self, config, refer, device, errors, collate_fn):
        super().__init__()
        
        self.refer = refer
        self.errors = errors
        self.collate_fn = collate_fn
        self.device = device,
        self.data_dir = _config["data_root"]

        self.num_workers = _config["num_workers"]
        self.batch_size = _config["per_gpu_batchsize"]
        self.eval_batch_size = self.batch_size

        self.image_size = _config["image_size"]
        self.max_text_len = _config["max_text_len"]
        self.draw_false_image = _config["draw_false_image"]
        self.draw_false_text = _config["draw_false_text"]
        self.image_only = _config["image_only"]

        self.train_transform_keys = (
            ["default_train"]
            if len(_config["train_transform_keys"]) == 0
            else _config["train_transform_keys"]
        )

        self.val_transform_keys = (
            ["default_val"]
            if len(_config["val_transform_keys"]) == 0
            else _config["val_transform_keys"]
        )

        tokenizer = _config["tokenizer"]
        # This is not adaptable, create function to accomodate changes in model
        self.tokenizer = ElectraTokenizer.from_pretrained(tokenizer)
        self.vocab_size = self.tokenizer.vocab_size

        
    def set_train_dataset(self):
        self.train_dataset = RefcocoDataset(
            self.refer, 
            self.tokenizer,
            self.device,
            self.errors,
            split='train'
        )

    def set_val_dataset(self):
        self.val_dataset = RefcocoDataset(
            self.refer, 
            self.tokenizer,
            self.device,
            self.errors,
            split='val'
        )
        
    def setup(self, stage: str):
        self.set_train_dataset()
        self.set_val_dataset()

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )
        
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )
        
        return loader
    
config = copy.deepcopy(_config)
pl.seed_everything(_config["seed"])
model = METERTransformerSS(config)
model.current_tasks = ['ref']

errors_df = pd.read_csv('Errors.csv')
errors_list = errors_df['Sent ID'].to_list()
dm = RefcocoDataModule(config, refer, device, errors_list, collate)

pl.seed_everything(_config["seed"])

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
    gpus=1,
    num_nodes=_config["num_nodes"],
    precision=_config["precision"],
#     accelerator="ddp",
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

# log_dir = logger.log_dir
# eval_file = 'eval.txt'
# eval_path = os.path.join(log_dir, eval_file )
# setattr(model, f"eval_path", eval_path)
# f = open(eval_path,'w') 
# f.close()

if not _config["test_only"]:
    trainer.fit(model, datamodule=dm)
else:
    trainer.test(model, datamodule=dm)










