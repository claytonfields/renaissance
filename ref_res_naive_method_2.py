#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 12:53:12 2023

@author: claytonfields
"""


import copy
import pytorch_lightning as pl
from tqdm import tqdm
import numpy as np
from refer import REFER
import pandas as pd
import torch
from torch.optim import AdamW

from transformers import ElectraTokenizer

from refcoco_utils_naive_method_2 import _config
from refcoco_utils_naive_method_2 import RefcocoDataset

from meter.modules import METERTransformerSS
from meter.datamodules import VQAv2DataModule



# Training Loop Function
def train(model, training_ds, optimizer, loss_fn, device):
    model.to(device)
    model.train()
    losses = []
    for data in tqdm(training_ds):
        if data['sent_id'] in training_ds.duds:
            continue
        try:
            optimizer.zero_grad()
            
            sent_id = data['sent_id']
            text = data['text']
            text_ids = data['text_ids']
            text_labels = data['text_labels']
            text_masks = data['text_masks']
            features = []
            for i,sub_image in enumerate(data['image']):
                
                ### TODO: Put all of the sub images in the infer dict with the coressponding sentence.
                input_dict = {
                    'image' : [sub_image],
                    'text' : text,
                    'text_ids' : text_ids,
                    'text_labels' : text_labels,
                    'text_masks' : text_masks
                }
                infer_dict = model.infer(input_dict)
        
                features.append(infer_dict['cls_feats'])
            
            cls_tensor = torch.cat(features)
            logits = model.ref_classifier(cls_tensor)

            obj_ids = data['obj_ids']
            ann_id = data['ann_id']

            target = torch.tensor([obj_ids.index(ann_id)]).to(device)
            loss = loss_fn(logits.reshape(1,-1),target)
            losses.append(loss.item())
            loss.backward()

            optimizer.step()
        except RuntimeError:
            print(f'Runtime Error at sent_id = {sent_id}')
            training_ds.duds.append(sent_id)
    return losses, loss

# Eval Loop Function
def evaluate(model, eval_ds):
    gold = []
    with torch.no_grad():
        for data in tqdm(eval_ds):
            if data['sent_id'] in eval_ds.duds:
                continue
            try:
                sent_id = data['sent_id']
                text = data['text']
                text_ids = data['text_ids']
                text_labels = data['text_labels']
                text_masks = data['text_masks']
                
                features = []
                for i,sub_image in enumerate(data['image']):
    
                    ### TODO: Put all of the sub images in the infer dict with the coressponding sentence.
                    input_dict = {
                        'image' : [sub_image],
                        'text' : text,
                        'text_ids' : text_ids,
                        'text_labels' : text_labels,
                        'text_masks' : text_masks
                    }
                    infer_dict = model.infer(input_dict)
                    features.append(infer_dict['cls_feats'])
                    
                cls_tensor = torch.cat(features)
                logits = model.ref_classifier(cls_tensor)
                
                obj_ids = data['obj_ids']
                ann_id = data['ann_id']
    
                pred_index = logits.argmax()
                pred_id = obj_ids[pred_index]
                if pred_id == ann_id:
                    gold.append(1)
                else:
                    gold.append(0)
            except RuntimeError:
                print(f'RuntimeError at sent_id = {sent_id}')
                eval_ds.duds.append(sent_id)
    return gold

        
# def main():
# data_root = '/home/claytonfields/nlp/code/data/coco'  # contains refclef, refcoco, refcoco+, refcocog and images
data_root = '/data/clayton/datasets/coco'
dataset = 'refcoco' 
splitBy = 'unc'
refer = REFER(data_root, dataset, splitBy)

# splits = ['train', 'val', 'test']
# refer.IMAGE_DIR = '/home/claytonfields/nlp/code/data/coco/images/mscoco/train2014'

config = copy.deepcopy(_config)
pl.seed_everything(config["seed"])

model = METERTransformerSS(config)


# Ref Res with METER
tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
optimizer = AdamW(model.parameters(), lr=1e-4)
loss_fn = torch.nn.functional.cross_entropy
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')

epochs = 2
BATCH_SIZE = 1

# Training Data
train_ds = RefcocoDataset(refer, tokenizer, device, split='train')
# train_params = {'batch_size': BATCH_SIZE,
#                 'shuffle': False,
#                 'num_workers': 0
#                 }
# training_loader = torch.utils.data.DataLoader(train_ds, **train_params)

# Eval Data
eval_ds = RefcocoDataset(refer, tokenizer, device, split='val')
# eval_params = {'batch_size': BATCH_SIZE,
#                 'shuffle': True,
#                 'num_workers': 0
#                 }
# eval_loader = torch.utils.data.DataLoader(eval_ds, **eval_params)
        
with open('eval.txt','w') as f:
    for epoch in range(epochs):
        losses, loss = train(model, train_ds, optimizer, loss_fn, device)
        avg_loss = np.average(losses)
        loss_string = f'Epoch: {epoch}, Final Loss: {loss.item()}, Average Loss: {avg_loss} \n'
        f.write(loss_string)
        print(loss_string)  
        pd.DataFrame(losses, columns=['Loss']).to_csv(f'Epoch_{epoch}_losses.csv')
        
        gold = evaluate(model, eval_ds)
        acc = np.average(gold)
        acc_string = f'Epoch: {epoch}, Acurracy on test set: {acc} \n'
        f.write(acc_string)
        print(acc_string)
pd.DataFrame(train_ds.duds, columns=['Sent ID']).to_csv('TrainingErrors.csv')
pd.DataFrame(eval_ds.duds, columns=['Sent ID']).to_csv('EvalErrors.csv')

# if __name__ == "__main__":
    # main()






































