#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 12:34:37 2023

@author: claytonfields

Dataset class for refCOCO 
"""

import pandas as pd
from .base_dataset import BaseDataset
from  refer import REFER, get_bounded_subimage
import io
from PIL import Image
import torch

refer_root = "/home/claytonfields/nlp/code/data/coco"
dataset = 'refcoco' 
splitBy = 'unc'
refer = REFER(refer_root, dataset, splitBy)

class RefcocoDataset(torch.utils.data.Dataset):

    def __init__(self, tokenizer, split='', max_bb = 42):
        self.tokenizer = tokenizer
        self.refer = refer
        self.max_bb = max_bb
        # self.device = device
        errors_df = pd.read_csv('Errors.csv')
        errors_list = errors_df['Sent ID'].to_list()
        self.errors = errors_list
        self.split = split
        self.sent_ids = self.get_sent_ids()
        self.duds = []

    def __len__(self):
        return len(self.sent_ids)
    
    def get_sent_ids(self):
        sent_ids = []
        for ref_id in self.refer.getRefIds(split=self.split):
            
            ref = self.refer.Refs[ref_id]
            img_id = ref['image_id']
            objs = refer.imgToAnns[img_id]
            if len(objs) <= self.max_bb:
                for sent_id in ref['sent_ids']:
                    if not sent_id in self.errors:
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
        ids = self.tokenizer.encode(
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
        
        target = torch.tensor([obj_ids.index(ann_id)])

        return_dict = {
            'ann_id' : ann_id,
            'image' : [torch.cat(sub_images)],#.to(self.device)],
            'obj_ids' : torch.tensor(obj_ids_total),#.to(self.device),
            'target' : target,#.to(self.device),
            'text' : sent['sent'],
            'text_ids' : text_ids,#.to(self.device),
            'text_labels' : text_labels,#.to(self.device),
            'text_masks' : text_masks,#.to(self.device)
        }
        
        return return_dict













