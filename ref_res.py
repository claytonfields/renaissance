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

from meter.transforms import keys_to_transforms
from meter.config import ex
from meter.modules import METERTransformerSS
from meter.datamodules.multitask_datamodule import MTDataModule
from meter.datasets.base_dataset import BaseDataset

data_root = '/home/claytonfields/nlp/code/data/coco'  # contains refclef, refcoco, refcoco+, refcocog and images
dataset = 'refcoco' 
splitBy = 'unc'
refer = REFER(data_root, dataset, splitBy)

# print stats about the given dataset
print ('dataset [%s_%s] contains: ' % (dataset, splitBy))
ref_ids = refer.getRefIds()
image_ids = refer.getImgIds()
print ('%s expressions for %s refs in %s images.' % (len(refer.Sents), len(ref_ids), len(image_ids)))

print ('\nAmong them:')
if dataset == 'refclef':
    if splitBy == 'unc':
        splits = ['train', 'val', 'testA', 'testB', 'testC']
    else:
        splits = ['train', 'val', 'test']
elif dataset == 'refcoco':
    splits = ['train', 'val', 'test']
elif dataset == 'refcoco+':
    splits = ['train', 'val', 'test']
elif dataset == 'refcocog':
    splits = ['train', 'val']  # we don't have test split for refco

refer.IMAGE_DIR = '/home/claytonfields/nlp/code/data/coco/images/mscoco/train2014'
# ex = Experiment("METER")

_config = copy.deepcopy(_config)
pl.seed_everything(_config["seed"])

dm = MTDataModule(_config, dist=False)
model = METERTransformerSS(_config)
exp_name = f'{_config["exp_name"]}'


# dm.prepare_data()
# dm.setup('fit')

# epochs = 1
# loader = dm.train_dataloader()
optim = AdamW(model.parameters(), lr=1e-4)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Ref Res with METER
tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')



epochs = 1
# loader = dm.train_dataloader()
optim = AdamW(model.parameters(), lr=1e-4)
loss_fn = torch.nn.functional.cross_entropy
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# Create loop for ref res
train_ids = refer.getRefIds(split='train')
text_labels = [[-100 for i in range(40)]]
# train_ids = train_ids[:5]

gold = []
model.train()
for ref_id in tqdm(train_ids):
    ref = refer.Refs[ref_id]
    img_id = ref['image_id']
    ann_id = ref['ann_id']
    objs = refer.imgToAnns[img_id]
    obj_ids = [obj['id'] for obj in objs]
    
    sub_images = []
    for obj in objs:
        x_a = get_bounded_subimage(refer, img_id, obj['id'], xs=224,ys=224, show=False)
        if x_a is not None:
            sub_images.append(x_a)
    num_sub_images = len(sub_images)
        
    
    for sent in ref['sentences']:
        scores = []
        for sub_image in sub_images:
            text_ids = tokenizer.encode(
                sent['sent'],
                padding="max_length",
                truncation=True,
                max_length=40,
                return_special_tokens_mask=True,
            )
            text_masks = torch.tensor([1 if text_ids[i]>0 else 0 for i,_ in enumerate(text_ids)]).reshape(1,-1)
            optim.zero_grad()
            
            ### TODO: Put all of the sub images in the infer dict with the coressponding sentence.
          
            input_dict = {
                'image' : [sub_image],
                'text' : sent,
                'text_ids' : torch.tensor(text_ids).reshape(1,-1),
                'text_labels' : text_labels,
                'text_masks' : text_masks
            }
            infer_dict = model.infer(input_dict)
            score = model.ref_classifier(infer_dict['cls_feats'])
            scores.append(score)
        # else:
        #     scores.append(0)
        # # pred_index = np.argmax(scores)
        # pred_id = objs[pred_index]['id']
        target = torch.tensor([obj_ids.index(ann_id)])
        scores = torch.cat(scores)
        loss = loss_fn(scores.reshape(1,-1),target)
        loss.backward()

        # Adjust learning weights
        optim.step()

## Eval Loop
eval_ids = refer.getRefIds(split='val')
with torch.no_grad():
    gold = []
    for ref_id in tqdm(eval_ids):
        ref = refer.Refs[ref_id]
        img_id = ref['image_id']
        ann_id = ref['ann_id']
        objs = refer.imgToAnns[img_id]
        obj_ids = [obj['id'] for obj in objs]
    
        sub_images = []
        for obj in objs:
            x_a = get_bounded_subimage(refer, img_id, obj['id'], xs=224,ys=224, show=False)
            if x_a is not None:
                sub_images.append(x_a)
        num_sub_images = len(sub_images)
        for sent in ref['sentences']:
            scores = []
            text_ids = tokenizer.encode(
                sent['sent'],
                padding="max_length",
                truncation=True,
                max_length=40,
                return_special_tokens_mask=True,
            )
            text_masks = torch.tensor([1 if text_ids[i]>0 else 0 for i,_ in enumerate(text_ids)]).reshape(1,-1)
    
            for sub_image in sub_images:
                # assert not torch.isnan(x_a).any()
                input_dict = {
                    'image' : [sub_image],
                    'text' : sent,
                    'text_ids' : torch.tensor(text_ids).reshape(1,-1),
                    'text_labels' : text_labels,
                    'text_masks' : text_masks
                }
                infer_dict = model.infer(input_dict)
                score = model.ref_classifier(infer_dict['cls_feats'])
                scores.append(score)

            pred_index = np.argmax(scores)
            pred_id = objs[pred_index]['id']
            target = torch.tensor([obj_ids.index(ann_id)])
            scores = torch.cat(scores)
            if pred_id == ann_id:
                gold.append(1)
            else:
                gold.append(0)
            
        
        
            
            















































