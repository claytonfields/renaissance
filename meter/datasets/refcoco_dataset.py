#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 12:34:37 2023

@author: claytonfields

Dataset class for refCOCO 
"""


from .base_dataset import BaseDataset
from  refer import REFER, get_bounded_subimage
import io
from PIL import Image
import torch

data_root = '/home/claytonfields/nlp/code/data/coco'  # contains refclef, refcoco, refcoco+, refcocog and images
dataset = 'refcoco' 
splitBy = 'unc'
refer = REFER(data_root, dataset, splitBy)

class RefCocoDataset(BaseDataset):
    def __init__(self, data_root, tokenizer, max_bb = 75):
        
        self.data_root = data_root
        self.dataset = 'refcoco'
        self.splitBy = 'unc'
        self.refer = REFER(data_root, dataset, splitBy)
        
        self.tokenizer = tokenizer
        self.max_bb = max_bb
        self.sent_ids = self.get_sent_ids()
        
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
        sent = refer.Sents[index]
        ref = refer.sentToRef[index]
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
        text_ids = torch.concat((repeat_ids, pad_ids)).to(torch.int)
        # text masks
        num_tokens = torch.where(text_ids[0] > 0)[0].size(dim=0)
        masks = torch.concat((torch.ones(num_tokens), torch.zeros(40-num_tokens))).to(torch.int)
        repeat_masks = masks.repeat(num_sub_images,1)
        pad_masks = torch.zeros(num_pad, 40)
        text_masks = torch.concat((repeat_masks, pad_masks)).to(torch.int)
        # text_labels
        labels = torch.full((40,),-100)
        repeat_labels = labels.repeat(num_sub_images, 1)
        pad_labels = torch.zeros(num_pad, 40)
        text_labels = torch.concat((repeat_labels, pad_labels)).to(torch.int)

        return_dict = {
            'ann_id' : ann_id,
            'image' : sub_images,
            'obj_ids' : torch.tensor(obj_ids_total),
            'text' : sent['sent'],
            'text_ids' : text_ids,
            'text_labels' : text_labels,
            'text_masks' : text_masks
        }
        
        return return_dict













