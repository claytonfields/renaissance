#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 16:15:32 2024

@author: claytonfields
"""


import copy
import os

import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoImageProcessor
from datasets import load_dataset

from meter.modules import METERTransformerSS
from meter.datamodules.multitask_datamodule import MTDataModule
from meter.datasets.base_dataset import BaseDataset
from meter.datasets.refcoco_dataset import RefcocoDataset
from meter.datamodules.datamodule_base import BaseDataModule


config = {  
    "exp_name":"finetune_mrpc",
    "seed" : 42,
    # "datasets" : ["coco", "vg", "sbu", "gcc"],
    "datasets" : ["coco", "vg"],
    # "datasets" : ["coco"],
    "loss_names" : {'itm': 0,
    'mlm': 0,
    'mpp': 0,
    'vqa': 0,
    'vcr': 0,
    'vcr_qar': 0,
    'nlvr2': 0,
    'irtr': 0,
    'contras': 0,
    'snli': 0,
    'ref': 0,
    'mrpc': 0,
    'rte' : 0,
    'wnli': 0,
    'sst2' : 0,
    'qqp' : 0,
    'qnli' : 0,
    'mnli' : 0,
    'cola' : 1
    },
    "batch_size" : 32,  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.
    "eval_batch_size" : 32,
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
    "max_text_len" : 128,

    # CrossLayer Setting
    "num_cross_layers" : 6,
    "cross_layer_hidden_size" : 256,
    "num_cross_layer_heads" : 4,
    "cross_layer_mlp_ratio" : 4,
    "cross_layer_drop_rate" : 0.1,
    
    # Architecture Setting
    "two_tower" : False,
    "multi_modal_encoder" : 'dandelin/vilt-b32-mlm',
    
    
    
    # Optimizer Setting
    "optim_type" : "adamw",
    "learning_rate" : 5e-5,
    "weight_decay" : 0.0,
    "decay_power" : 1,
    "max_epoch" : 3,
    "max_steps" : 100000,
    "warmup_steps" : 0,
    "end_lr" : 0,
    "lr_mult_head" : 5,  # multiply lr for downstream heads
    "lr_mult_cross_modal" : 5,  # multiply lr for the cross-modal module

    # Encoder Settings
    "freeze_image_encoder" : True,
    "freeze_text_encoder" : False,
    'freeze_cross_modal_layers' : True,
    
    'text_only' : False,
    

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
    "per_gpu_batchsize" : 32,  # you should define this manually with per_gpu_batch_size:#
    "num_gpus" : 1,
    "num_nodes" : 1,
    "load_path" : "/home/claytonfields/nlp/code/meter/result/mlm_itm_seed0_from_/meter_electra_small_deit_tiny_p16_is224_bs288_is1M/checkpoints/epoch=43-step=898039.ckpt",
    # "load_path" : '/home/claytonfields/nlp/code/meter/result/mlm_itm_deit_fr_electra_fr_is224_ps16_bs336_pgbs84_ts100k/checkpoints/epoch=5-step=96215.ckpt',
    "num_workers" : 12,
    "precision" : 32
}





# class GlueDataset(torch.utils.data.Dataset):
#     def __init__(self, task, split, tokenizer, max_length=128):

#         self.task = task
#         self.tasks = ["cola","mnli","mrpc","qnli","qqp","rte","sst2","stsb","wnli"]
#         if self.task not in self.tasks:
#             raise ValueError("The selected GLUE task is not supported.")
#         self.split = split
#         self.tokenizer = tokenizer
#         self.max_length = max_length

#         self.data_dict = load_dataset('glue', self.task, split=self.split).to_dict()
#         if self.task in ["rte", "mrpc", "stsb", "wnli"]:
#             self.sentence1 = self.data_dict['sentence1']
#             self.sentence2 = self.data_dict['sentence2']
#         elif self.task in ["cola", "sst2"]:
#             self.sentence1 = self.data_dict['sentence']
#             self.sentence2 = None
#         elif self.task in ["qqp"]:
#             self.sentence1 = self.data_dict["question1"]
#             self.sentence2 = self.data_dict["question2'"]
#         elif self.task in ["qnli"]:
#             self.sentence1 = self.data_dict["question"]
#             self.sentence2 = self.data_dict["sentence"]
#         elif self.task in ["mnli"]:
#             self.sentence1 = self.data_dict["premise"]
#             self.sentence2 = self.data_dict["hypothesis"]
#         self.label = self.data_dict['label']
#         if self.task == "cola":
#             self.idx = self.data_dict["idx"]
#         else:
#             self.idx = self.data_dict['idx']

#     def __len__(self):
#         return len(self.idx)

#     def __getitem__(self, index):

#         sent1 = self.sentence1[index]
#         if self.sentence2:
#             sent2 = self.sentence2[index]
#         else:
#             sent2 = None
#         label = self.label[index]
#         # idx = self.idx[index]

#         ret = self.tokenizer(
#             sent1, 
#             sent2,
#             max_length=self.max_length,
#             padding='max_length',
#             truncation=True,
#             return_tensors='pt'
#         )
#         ret = {k: v.squeeze() for k,v in ret.items()}
#         ret['label'] = label
#         return ret

# class GlueDataset(BaseDataset):
#     def __init__(self,*args,  task='', split='', max_text_length=128, **kwargs):

#             self.task = task
#             self.tasks = ["cola","mnli","mrpc","qnli","qqp","rte","sst2","stsb","wnli"]
#             if self.task not in self.tasks:
#                 raise ValueError("The selected GLUE task is not supported.")
#             # self.data_dict = load_dataset('glue', self.task, split=self.split).to_dict()
            
            
#             if  split not in ["train", "val", "test"]:
#                 raise ValueError(f"{split} is not a recognized data split.")
#             self.split = split
#             # self.tokenizer = tokenizer
#             self.max_text_length = max_text_length
            
#             super().__init__(*args, [], text_column_name="", **kwargs)
            
#             if self.task in ["rte", "mrpc", "stsb", "wnli"]:
#                 self.sentence1 = self.data_dict['sentence1']
#                 self.sentence2 = self.data_dict['sentence2']
#             elif self.task in ["cola", "sst2"]:
#                 self.sentence1 = self.data_dict['sentence']
#                 self.sentence2 = None
#             elif self.task in ["qqp"]:
#                 self.sentence1 = self.data_dict["question1"]
#                 self.sentence2 = self.data_dict["question2'"]
#             elif self.task in ["qnli"]:
#                 self.sentence1 = self.data_dict["question"]
#                 self.sentence2 = self.data_dict["sentence"]
#             elif self.task in ["mnli"]:
#                 self.sentence1 = self.data_dict["premise"]
#                 self.sentence2 = self.data_dict["hypothesis"]
#             self.label = self.data_dict['label']
#             self.idx = self.data_dict["idx"]
            
#     def __len__(self):
#             return len(self.idx)

#     def __getitem__(self, index):

#         sent1 = self.sentence1[index]
#         if self.sentence2:
#             sent2 = self.sentence2[index]
#         # else:
#         #     sent2 = None
#         label = self.label[index]
#         # idx = self.idx[index]

#         ret = self.tokenizer(
#             sent1, 
#             sent2,
#             max_length=self.max_text_length,
#             padding='max_length',
#             truncation=True,
#             return_tensors='pt'
#         )
#         ret = {k: v.squeeze() for k,v in ret.items()}
#         ret['label'] = label
#         return ret

    
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
processor = AutoImageProcessor.from_pretrained("facebook/deit-tiny-patch16-224")

data_dir = '/home/claytonfields/nlp/code/meter/data/arrow'
transform_keys = ['imagenet']
image_size = 32

from meter.datasets.glue_dataset import GlueDataset

ds_train = GlueDataset(
    # hugging_face=True, 
    hf_dataset_key='glue',
    task='mrpc',
    split='train', 
    tokenizer=tokenizer
)
ds_ref_train = RefcocoDataset(
    data_dir=data_dir,
    transform_keys=transform_keys,
    image_size=image_size,
    tokenizer=tokenizer, 
    processor=processor, 
    split='train'
)
# ds = GlueDataset('mrpc', 'train', tokenizer)
# data_dict = ds.data_dict
# ds_test = RefcocoDataset(data_dir, transform_keys, image_size, tokenizer=tokenizer, processor=processor, split='test')
# ds_val = RefcocoDataset(data_dir, transform_keys, image_size, tokenizer=tokenizer, processor=processor, split='val')
data = ds_train[0]







# class GlueDataModule(BaseDataModule):
#     def __init__(self, *args, **kwargs):

#         super().__init__(*args, **kwargs)
        
#         self.hf_dataset_key = "glue"
        
#         if config["loss_names"]["snli"] > 0:
#             self.task = "snli"
#         elif config["loss_names"]["mrpc"] > 0:
#             self.task = "mrpc"
#         elif config["loss_names"]["rte"] > 0:
#             self.task = "rte" 
#         elif config["loss_names"]["wnli"] > 0:
#             self.task = "wnli"
#         elif config["loss_names"]["sst2"] > 0:
#             self.task = "sst2"
#         elif config["loss_names"]["qqp"] > 0:
#             self.task = "qqp"
#         elif config["loss_names"]["qnli"] > 0:
#             self.task = "qnli"
#         elif config["loss_names"]["mnli"] > 0:
#             self.task = "mnli"
#         elif config["loss_names"]["cola"] > 0:
#             self.task = "cola"
#         else:
#             raise ValueError("Selected task is not supported by GlueDataModule.")
        
#     @property
#     def dataset_cls(self):
#         return GlueDataset

#     @property
#     def dataset_cls_no_false(self):
#         return GlueDataset
#     @property
#     def dataset_name(self):
#         return "glue"

from meter.datamodules.glue_datamodule import GlueDataModule

# dm = RefcocoDataModule(config)
dm = GlueDataModule(config)
dm.setup('train')
dl = dm.train_dataloader()
batch = next(iter(dl))
# dm.prepare_data()
# dl = dm.test_dataloader()
# batch = next(iter(dl))


# ds = GlueDataset('mrpc', 'train', tokenizer)
# dl = DataLoader(ds, batch_size=10)
# batch = next(iter(dl))

# config = copy.deepcopy(_config)
# print(config)
# pl.seed_everything(_config["seed"])
# model = METERTransformerSS(config)
# model.current_tasks = ['mrpc']
# encoder = model.encoder
































