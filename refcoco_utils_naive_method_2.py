#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 14:05:13 2023

@author: claytonfields

"""

# from refer import REFER
import io
import os
from PIL import Image

import numpy as np
import skimage.io as skio
# import matplotlib.pyplot as plt

from torchvision import transforms
import torch
# import refer


# data_root = '/home/claytonfields/nlp/code/data/coco'  # contains refclef, refcoco, refcoco+, refcocog and images
# dataset = 'refcoco' 
# splitBy = 'unc'
# refer = REFER(data_root, dataset, splitBy)

def get_bounded_subimage(refer, img_id, ann_id, xs=224,ys=224, show=False):
    bbox = refer.Anns[ann_id]['bbox']
    bbox = [int(b) for b in bbox]
    img = refer.Imgs[img_id]
    I = skio.imread(os.path.join(refer.IMAGE_DIR, img['file_name']))
    sub = I[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
    if show:
        plt.figure()
        ax = plt.gca()
        ax.imshow(sub)
        plt.show()
        pass
    if len(sub) == 0: return None
    pim = Image.fromarray(sub)
    pim2 = pim.resize((xs,ys), Image.ANTIALIAS)
    # img = np.array(pim2)
    img = transforms.functional.to_tensor(pim2)
    if len(img.shape) < 3: return None
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    return img


def compute_posfeats(refer, img_id, ann_id,):
    img = refer.Imgs[img_id]
    bb = refer.Anns[ann_id]['bbox']
    fname = os.path.join(refer.IMAGE_DIR, img['file_name'])
    if not os.path.isfile(fname): return None
    img = io.imread(fname)
    
    if len(img.shape) < 3: return None
    ih, iw, _ = img.shape
    x,y,w,h = bb
    # x1, relative
    x1r = x / iw
    # y1, relative
    y1r = y / ih
    # x2, relative
    x2r = (x+w) / iw
    # y2, relative
    y2r = (y+h) / ih
    # area
    area = (w*h) / (iw*ih)
    # ratio image sides (= orientation)
    ratio = iw / ih
    # distance from center (normalised)
    cx = iw / 2
    cy = ih / 2
    bcx = x + w / 2
    bcy = y + h / 2
    distance = np.sqrt((bcx-cx)**2 + (bcy-cy)**2) / np.sqrt(cx**2+cy**2)
    # done!
    return np.array([x1r,y1r,x2r,y2r,area,ratio,distance]).reshape(1,7)

def _loss_names(d):
    ret = {
        "itm": 0,
        "mlm": 0,
        "mpp": 0,
        "vqa": 1,
        "vcr": 0,
        "vcr_qar": 0,
        "nlvr2": 0,
        "irtr": 0,
        "contras": 0,
        "snli": 1,
        "ref": 1
    }
    ret.update(d)
    return ret

# RefCOCO data class
class RefcocoDataset(torch.utils.data.Dataset):

    def __init__(self, refer, tokenizer, device, split='', max_bb = 75):
        self.tokenizer = tokenizer
        self.refer = refer
        self.max_bb = max_bb
        self.split = split
        self.sent_ids = self.get_sent_ids()
        self.duds = []
        self.device = device
        

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
        
        img_id = ref['image_id']
        ann_id = ref['ann_id']
        objs = self.refer.imgToAnns[img_id]
        obj_ids = [obj['id'] for obj in objs]
        
        sub_images = []
        for obj in objs:
            try:
                x_a = get_bounded_subimage(self.refer, img_id, obj['id'], xs=224,ys=224, show=False)
            except ValueError:
                print(f'ValueError at setence id: {sent_id}')
                self.duds.append(sent_id)
                break
                
            if x_a is not None:
                sub_images.append(torch.tensor(x_a).to(self.device))
        num_sub_images = len(sub_images)      
            
        text_ids = self.tokenizer.encode(
            sent['sent'],
            padding="max_length",
            truncation=True,
            max_length=40,
            return_special_tokens_mask=True,
        )
        text_masks = [1 if text_ids[i]>0 else 0 for i,_ in enumerate(text_ids)]
        text_labels = [[-100 for i in range(40)]]

        # ids = [text_ids for i in range(num_sub_images)]
        # masks = [text_masks for _ in range(num_sub_images)]
        # labels = [text_labels for i in range(num_sub_images)]
            
        return_dict = {
            'ann_id' : ann_id,
            'image' : sub_images,
            'obj_ids' : obj_ids,
            'sent_id' : sent_id,
            'text' : sent['sent'],
            'text_ids' : torch.tensor(text_ids).reshape(1,-1).to(self.device),
            'text_labels' : torch.tensor(text_labels).reshape(1,-1).to(self.device),
            'text_masks' : torch.tensor(text_masks).reshape(1,-1).to(self.device)
        }  

        return return_dict



_config = {  
    "exp_name":"meter",
    "seed" : 0,
    # "datasets" : ["coco", "vg", "sbu", "gcc"],
    # "datasets" : ["coco", "vg"],
    "datasets" : ["coco"],
    "loss_names" : _loss_names({"itm": 1, "mlm": 1}),
    "batch_size" : 1,  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.

    # Image setting
    "train_transform_keys" : ["imagenet"],
    "val_transform_keys" : ["imagenet"],
    "image_size" : 224,
    "patch_size" : 16,
    "draw_false_image" : 1,
    "image_only" : False,
    "resolution_before" : 224,

    # Text Setting
    "vqav2_label_size" : 3129,
    "max_text_len" : 40,
    "tokenizer" : "google/electra-small-discriminator",
    "vocab_size" : 30522,
    "whole_word_masking" : False, # note that whole_word_masking does not work for RoBERTa
    "mlm_prob" : 0.15,
    "draw_false_text" : 0,

    # Transformer Setting
    "num_top_layer" : 6,
    "input_image_embed_size" : 192,
    "input_text_embed_size" : 256,
    "vit" : "vit_deit_tiny_patch16_224",
    "hidden_size" : 256,
    "num_heads" : 4,
    "num_layers" : 6,
    "mlp_ratio" : 4,
    "drop_rate" : 0.1,

    # Optimizer Setting
    "optim_type" : "adamw",
    "learning_rate" : 1e-5,
    "weight_decay" : 0.01,
    "decay_power" : 1,
    "max_epoch" : 100,
    "max_steps" : 100000,
    "warmup_steps" : 10000,
    "end_lr" : 0,
    "lr_mult_head" : 5,  # multiply lr for downstream heads
    "lr_mult_cross_modal" : 5,  # multiply lr for the cross-modal module

    # Downstream Setting
    "get_recall_metric" : False,
    
    
    "model_type" : "METER",

    # PL Trainer Setting
    "resume_from" : None,
    "fast_dev_run" : False,
    "val_check_interval" : 1.0,
    "test_only" : False,

    # below params varies with the environment
    "data_root" : "/home/claytonfields/nlp/code/vilt/data/arrow",
    "log_dir" : "result",
    "per_gpu_batchsize" : 1,  # you should define this manually with per_gpu_batch_size:#
    "num_gpus" : 1,
    "num_nodes" : 1,
    "load_path" : "/home/claytonfields/nlp/code/meter/result/mlm_itm_seed0_from_/meter_electra_small_deit_tiny_p16_is224_bs288_is1M/checkpoints/epoch=43-step=898039.ckpt",
    "num_workers" : 12,
    "precision" : 32
}
