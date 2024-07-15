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
from meter.datamodules import VQAv2DataModule
from meter.datasets.base_dataset import BaseDataset


class NewRefcocoDataset(torch.utils.data.Dataset):

    def __init__(self, refer, tokenizer, split='', max_bb = 40):
        self.tokenizer = tokenizer
        self.refer = refer
        self.max_bb = max_bb
        self.split = split
        self.sent_ids = self.get_sent_ids()
        self.duds = []

    def __len__(self):
        return len(self.sent_ids)
    
    def get_sent_ids(self):
        sent_ids = []
        for ref_id in self.refer.getRefIds():
            ref = self.refer.Refs[ref_id]
            for sent_id in ref['sent_ids']:
                sent_ids.append(sent_id)
        return sent_ids
    
    def __getitem__(self, index):
        max_bb = self.max_bb
        sent_id = self.sent_ids[index]
        ref = self.refer.sentToRef[sent_id]
        sent = self.refer.Sents[sent_id]
        
        img_id = ref['image_id']
        ann_id = ref['ann_id']
        objs = refer.imgToAnns[img_id]
        obj_ids = [obj['id'] for obj in objs]
        obj_pad = [0 for _ in range(max_bb-len(obj_ids))]
        obj_ids_total = obj_ids+obj_pad

        sub_images = []
        for obj in objs:
            x_a = get_bounded_subimage(refer, img_id, obj['id'], xs=224,ys=224, show=False)
            if x_a is not None:
                sub_images.append(x_a)
        
        num_sub_images = len(sub_images)
        num_pad = max_bb - num_sub_images 
        
        pad_image = torch.zeros(1,3,224,224)
        for _ in range(max_bb - num_sub_images):
            sub_images.append(pad_image)
        
        # text ids
        ids = tokenizer.encode(
            sent['sent'],
            padding="max_length",
            truncation=True,
            max_length=40,
            return_special_tokens_mask=True,
        )
        repeat_ids = torch.tensor(ids).repeat(num_sub_images,1)
        pad_ids =  torch.zeros(num_pad,40)
        text_ids = torch.cat((repeat_ids, pad_ids)).to(torch.long)
        # text masks
        num_tokens = torch.where(text_ids[0] > 0)[0].size(dim=0)
        masks = torch.cat((torch.ones(num_tokens), torch.zeros(40-num_tokens))).to(torch.long)
        repeat_masks = masks.repeat(num_sub_images,1)
        pad_masks = torch.zeros(num_pad, 40)
        text_masks = torch.cat((repeat_masks, pad_masks)).to(torch.long)
        # text_labels
        labels = torch.full((40,),-100)
        repeat_labels = labels.repeat(num_sub_images, 1)
        pad_labels = torch.zeros(num_pad, 40)
        text_labels = torch.cat((repeat_labels, pad_labels)).to(torch.long)

        return_dict = {
            'ann_id' : ann_id,
            'image' : sub_images,
            'obj_ids' : obj_ids_total,
            'sent_id' : sent_id,
            'text' : sent['sent'],
            'text_ids' : text_ids,
            'text_labels' : text_labels,
            'text_masks' : text_masks
        }
        
        return return_dict


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
optim = AdamW(model.parameters(), lr=1e-4)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Ref Res with METER
tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
BATCH_SIZE = 10


epochs = 1
# loader = dm.train_dataloader()
optim = AdamW(model.parameters(), lr=1e-4)
loss_fn = torch.nn.functional.cross_entropy
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

ds = NewRefcocoDataset(refer, tokenizer)
train_params = {'batch_size': BATCH_SIZE,
                'shuffle': False,
                'num_workers': 0
                }

training_loader = torch.utils.data.DataLoader(ds, **train_params)

for i, batch in enumerate(training_loader):
    if i==0:
        break
# batch['image'][0].shape
# data = ds[0]

# infer_dict = model.infer(data)
# cls_feats= infer_dict['cls_feats']
# logits = model.ref_classifier(infer_dict['cls_feats'])

# dm = VQAv2DataModule(_config)

dm.batch_size = 10

dm.prepare_data()
dm.setup('fit')
loader = dm.train_dataloader()

for i, data in enumerate(loader):
    if i ==1:
        break

model.current_tasks.append('vqa')
print(model.current_tasks)
output = model(data)
# model(ds[0])


# model.train()
# losses = []
# for data in tqdm(train_ds):
    
#     optim.zero_grad()

#     infer_dict = model.infer(data)
#     logits = model.ref_classifier(infer_dict['cls_feats'])

#     obj_ids = data['obj_ids']
#     ann_id = data['ann_id']
    
#     target = torch.tensor([obj_ids.index(ann_id)])
#     loss = loss_fn(logits.reshape(1,-1),target)
#     losses.append(loss.item())
#     loss.backward()

#     # Adjust learning weights
#     optim.step()


# ## Eval Loop
# eval_ids = refer.getRefIds(split='val')
# with torch.no_grad():
#     gold = []
#     for data in tqdm(train_ds):

#         infer_dict = model.infer(data)
#         logits = model.ref_classifier(infer_dict['cls_feats'])

#         obj_ids = data['obj_ids']
#         ann_id = data['ann_id']

#         pred_index = np.argmax(logits)
#         pred_id = obj_ids[pred_index]['id']
#         target = torch.tensor([obj_ids.index(ann_id)])
#         # scores = torch.cat(scores)
#         if pred_id == ann_id:
#             gold.append(1)
#         else:
#             gold.append(0)
            
        
        
            
            















































