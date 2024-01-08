#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 16:30:47 2023

@author: claytonfields
"""

from ..datasets import RefCocoDataset
from .datamodule_base import BaseDataModule

# TODO: incorporate error file directly into dataset
errors_df = pd.read_csv('Errors.csv')
errors_list = errors_df['Sent ID'].to_list()

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