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
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from datasets import load_dataset

from meter.modules import METERTransformerSS
# from meter.datamodules.multitask_datamodule import MTDataModule
# from meter.datasets.base_dataset import BaseDataset


_config = {  
    "exp_name":"finetune_mrpc",
    "seed" : 42,
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
    'ref': 0,
    'mrpc':1
    },
    "batch_size" : 32,  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.

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
    
    'text_only' : True,
    

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


class GlueDataset(torch.utils.data.Dataset):
    def __init__(self, task, split, tokenizer, max_length=128):

        self.task = task
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length

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

        ret = self.tokenizer(
            sent1, 
            sent2,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        ret = {k: v.squeeze() for k,v in ret.items()}
        ret['label'] = label
        return ret


class GlueDataModule(LightningDataModule):
    def __init__(self, config, task, batch_size=32, eval_batch_size=8):
        super().__init__()

        self.task = task
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(config['text_encoder'])
        
    def set_train_dataset(self):
        self.train_dataset = GlueDataset(self.task, 'train', self.tokenizer)

    def set_val_dataset(self):
        self.val_dataset =  GlueDataset(self.task, 'validation', self.tokenizer)

    def set_test_dataset(self):
         self.text_dataset = GlueDataset(self.task, 'test', self.tokenizer)
        
    def setup(self, stage: str):
        self.set_train_dataset()
        self.set_val_dataset()
        self.set_test_dataset()

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            # num_workers=self.num_workers,
            # pin_memory=True,
            # collate_fn=self.collate_fn,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            # num_workers=self.num_workers,
            # pin_memory=True,
            # collate_fn=self.collate_fn,
        )
        return loader
        
    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            # num_workers=self.num_workers,
            # pin_memory=True,
            # collate_fn=self.collate_fn,
        )
        return loader


# ds = GlueDataset('mrpc', 'train', tokenizer)
# dl = DataLoader(ds, batch_size=10)
# batch = next(iter(dl))

config = copy.deepcopy(_config)
print(config)
pl.seed_everything(_config["seed"])
model = METERTransformerSS(config)
model.current_tasks = ['mrpc']


dm = GlueDataModule(_config, 'mrpc')

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
    devices=num_gpus,
    num_nodes=_config["num_nodes"],
    precision=_config["precision"],
    # accelerator="ddp",
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
    # flush_logs_every_n_steps=10,
#     resume_from_checkpoint=_config["resume_from"],
    # weights_summary="top",
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






