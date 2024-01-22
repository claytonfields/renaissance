#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 16:30:47 2023

@author: claytonfields
"""

from pytorch_lightning import LightningDataModule
import pandas as pd
import torch
from torch.utils.data import DataLoader

# from ..datasets import RefCocoDataset
from .datamodule_base import BaseDataModule
from ..datasets.refcoco_dataset import RefcocoDataset

from transformers.models.auto import AutoTokenizer

# TODO: incorporate error file directly into dataset


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
        
        # errors_df = pd.read_csv('Errors.csv')
        # errors_list = errors_df['Sent ID'].to_list()
        # self.errors = errors_list
        self.collate_fn = collate_fn
        self.device = device,
        self.data_dir = config["data_root"]

        self.num_workers = config["num_workers"]
        self.batch_size = config["per_gpu_batchsize"]
        self.eval_batch_size = self.batch_size

        self.image_size = config["image_size"]
        self.max_text_len = config["max_text_len"]
        self.draw_false_image = config["draw_false_image"]
        self.draw_false_text = config["draw_false_text"]
        self.image_only = config["image_only"]

        self.train_transform_keys = (
            ["default_train"]
            if len(config["train_transform_keys"]) == 0
            else config["train_transform_keys"]
        )

        self.val_transform_keys = (
            ["default_val"]
            if len(config["val_transform_keys"]) == 0
            else config["val_transform_keys"]
        )

        tokenizer = config["tokenizer"]
        # This is not adaptable, create function to accomodate changes in model
        self.tokenizer = AutoTokenizer().from_pretrained(tokenizer)
        self.vocab_size = self.tokenizer.vocab_size

        
    def set_train_dataset(self):
        self.train_dataset = RefcocoDataset(
            self.refer, 
            # self.tokenizer,
            # self.device,
            self.errors,
            split='train'
        )

    def set_val_dataset(self):
        self.val_dataset = RefcocoDataset(
            self.refer, 
            self.tokenizer,
            # self.device,
            # self.errors,
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