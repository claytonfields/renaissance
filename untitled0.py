#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 16:02:21 2024

@author: claytonfields
"""

from meter.datamodules.refcoco_datamodule import RefcocoDataModule

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

dm = RefcocoDataModule(config)
dm.setup('train')
dl = dm.train_dataloader()
batch = next(iter(dl))