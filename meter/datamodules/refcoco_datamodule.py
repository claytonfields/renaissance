#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 16:30:47 2023

@author: claytonfields
"""

from ..datasets import RefCocoDataset
from .datamodule_base import BaseDataModule



class RefCocoDataModule(BaseDataModule):
    def __init__(self, data_dir: str = "", batch_size: int = 10):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: str):
        self.dataset = RefCocoDataModule(refer)

    @property
    def dataset_cls(self):
        return RefCocoDataset

    @property
    def dataset_cls_no_false(self):
        return RefCocoDataset

    @property
    def dataset_name(self):
        return "coco"
