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
# from refcoco_utils import RefcocoDataset

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

model = METERTransformerSS(_config)

# RefCOCO data class
class RefcocoDataset(torch.utils.data.Dataset):

    def __init__(self, refer, tokenizer, split='', max_bb = 75):
        self.tokenizer = tokenizer
        self.refer = refer
        self.max_bb = max_bb
        self.split = split
        self.sent_ids = self.get_sent_ids()[:50]
        

    def __len__(self):
        return len(self.sent_ids)
    
    def get_sent_ids(self):
        sent_ids = []
        for ref_id in self.refer.getRefIds(split=self.split):
            ref = self.refer.Refs[ref_id]
            for sent_id in ref['sent_ids']:
                sent_ids.append(sent_id)
        return sent_ids

    def __getitem__(self, index):
        sent_id = self.sent_ids[index]
        ref = self.refer.sentToRef[sent_id]
        sent = self.refer.Sents[sent_id]
        print(sent['sent_id'])
        
        img_id = ref['image_id']
        ann_id = ref['ann_id']
        objs = self.refer.imgToAnns[img_id]
        obj_ids = [obj['id'] for obj in objs]
        
        sub_images = []
        for obj in objs:
            x_a = get_bounded_subimage(refer, img_id, obj['id'], xs=224,ys=224, show=False)
            if x_a is not None:
                sub_images.append(x_a)
        num_sub_images = len(sub_images)      
            
        text_ids = tokenizer.encode(
            sent['sent'],
            padding="max_length",
            truncation=True,
            max_length=40,
            return_special_tokens_mask=True,
        )
        text_masks = [1 if text_ids[i]>0 else 0 for i,_ in enumerate(text_ids)]
        text_labels = [[-100 for i in range(40)]]

        ids = [text_ids for i in range(num_sub_images)]
        masks = [text_masks for _ in range(num_sub_images)]
        labels = [text_labels for i in range(num_sub_images)]
            
        return_dict = {
            'ann_id' : ann_id,
            'image' : sub_images,
            'obj_ids' : obj_ids,
            'sent_id' : sent_id,
            'text' : sent['sent'],
            'text_ids' : torch.tensor(ids),
            'text_labels' : torch.tensor(labels),
            'text_masks' : torch.tensor(masks)
        }  

        return return_dict


# Ref Res with METER
tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
optimizer = AdamW(model.parameters(), lr=1e-4)
loss_fn = torch.nn.functional.cross_entropy
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

epochs = 1
BATCH_SIZE = 1

# Training Data
train_ds = RefcocoDataset(refer, tokenizer, split='train')
train_params = {'batch_size': BATCH_SIZE,
                'shuffle': False,
                'num_workers': 0
                }
training_loader = torch.utils.data.DataLoader(train_ds, **train_params)

# Eval Data
eval_ds = RefcocoDataset(refer, tokenizer, split='val')
eval_params = {'batch_size': BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }
eval_loader = torch.utils.data.DataLoader(eval_ds, **eval_params)

# Training Loop Function
def train(model, training_ds, optimizer, loss_fn):
    model.train()
    losses = []
    for data in tqdm(training_ds):
        try:
            optimizer.zero_grad()
            
            sent_id = data['sent_id']
            infer_dict = model.infer(data)
            logits = model.ref_classifier(infer_dict['cls_feats'])

            obj_ids = data['obj_ids']
            ann_id = data['ann_id']

            target = torch.tensor([obj_ids.index(ann_id)])
            loss = loss_fn(logits.reshape(1,-1),target)
            losses.append(loss.item())
            loss.backward()

            optimizer.step()
        except RuntimeError:
            print(f'Runtime Error at sent_id = {sent_id}')
    return losses, loss

# Eval Loop Function
def evaluate(model, eval_ds):
    gold = []
    with torch.no_grad():
        for data in tqdm(eval_ds):
            try:
                sent_id = data['sent_id']
                infer_dict = model.infer(data)
                logits = model.ref_classifier(infer_dict['cls_feats'])

                obj_ids = data['obj_ids']
                ann_id = data['ann_id']

                pred_index = np.argmax(logits)
                pred_id = obj_ids[pred_index]#['id']
                if pred_id == ann_id:
                    gold.append(1)
                else:
                    gold.append(0)
            except:
                print(f'RuntimeError at sent_id = {sent_id}')
    return gold


for epoch in range(epochs):
    losses, loss = train(model, train_ds, optimizer, loss_fn)
    print(f'Epoch: {epoch}, Loss:  {loss.item()}')  
    gold = evaluate(model, eval_ds)
    acc = np.average(gold)
    print(f'acurracy on test set {acc}')
            
        
        
            
            















































