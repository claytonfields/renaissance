#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 12:53:12 2023

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
import matplotlib.pyplot as plt
from refer import REFER

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torchvision import transforms

from transformers import ElectraTokenizer

from refcoco_utils import get_bounded_subimage
from refcoco_utils import _config
from refcoco_utils import _loss_names
from refcoco_utils import RefcocoDataset

from meter.transforms import keys_to_transforms
from meter.config import ex
from meter.modules import METERTransformerSS
from meter.datamodules.multitask_datamodule import MTDataModule
from meter.datasets.base_dataset import BaseDataset

data_root = '/home/claytonfields/nlp/code/data/coco'  # contains refclef, refcoco, refcoco+, refcocog and images
dataset = 'refcoco' 
splitBy = 'unc'
refer = REFER(data_root, dataset, splitBy)
splits = ['train', 'val', 'test']

refer.IMAGE_DIR = '/home/claytonfields/nlp/code/data/coco/images/mscoco/train2014'
_config = copy.deepcopy(_config)
pl.seed_everything(_config["seed"])

dm = MTDataModule(_config, dist=False)
model = METERTransformerSS(_config)
exp_name = f'{_config["exp_name"]}'


# Ref Res with METER
train_ids = refer.getRefIds(split='train')
tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
BATCH_SIZE = 1
epochs = 1
optim = AdamW(model.parameters(), lr=1e-4)
loss_fn = torch.nn.functional.cross_entropy
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

ds = RefcocoDataset(refer, tokenizer)
train_ds = torch.utils.data.Subset(ds, train_ids)

train_params = {'batch_size': BATCH_SIZE,
                'shuffle': False,
                'num_workers': 0
}
training_loader = torch.utils.data.DataLoader(train_ds, **train_params)




model.train()
losses = []
for data in tqdm(train_ds):
    
    optim.zero_grad()

    infer_dict = model.infer(data)
    logits = model.ref_classifier(infer_dict['cls_feats'])

    obj_ids = data['obj_ids']
    ann_id = data['ann_id']
    
    target = torch.tensor([obj_ids.index(ann_id)])
    loss = loss_fn(logits.reshape(1,-1),target)
    losses.append(loss.item())
    loss.backward()

    # Adjust learning weights
    optim.step()


## Eval Loop
eval_ids = refer.getRefIds(split='val')
with torch.no_grad():
    gold = []
    for data in tqdm(train_ds):

        infer_dict = model.infer(data)
        logits = model.ref_classifier(infer_dict['cls_feats'])

        obj_ids = data['obj_ids']
        ann_id = data['ann_id']

        pred_index = np.argmax(logits)
        pred_id = obj_ids[pred_index]['id']
        target = torch.tensor([obj_ids.index(ann_id)])
        # scores = torch.cat(scores)
        if pred_id == ann_id:
            gold.append(1)
        else:
            gold.append(0)
            
        
        
            
            















































