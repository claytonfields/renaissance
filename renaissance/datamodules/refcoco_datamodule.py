#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 16:30:47 2023

@author: claytonfields
"""
from pytorch_lightning import LightningDataModule
import pandas as pd
from transformers import AutoTokenizer

import torch
from torch.utils.data import DataLoader

from ..datasets.refcoco_dataset import RefcocoDataset
# from  .refer import REFER
from .datamodule_base import BaseDataModule


# TODO: incorporate error file directly into dataset
# errors_df = pd.read_csv('Errors.csv')
# errors_list = errors_df['Sent ID'].to_list()



class RefcocoDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return RefcocoDataset

    @property
    def dataset_cls_no_false(self):
        return RefcocoDataset

    @property
    def dataset_name(self):
        return "refcoco"