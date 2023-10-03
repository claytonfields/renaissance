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

import torch
from torch.optim import AdamW

from transformers import ElectraTokenizer

from refcoco_utils_test import _config
from refcoco_utils_test import RefcocoDataset


from meter.modules import METERTransformerSS




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



            
        
        
def main():
    data_root = '/home/claytonfields/nlp/code/data/coco'  # contains refclef, refcoco, refcoco+, refcocog and images
    dataset = 'refcoco' 
    splitBy = 'unc'
    refer = REFER(data_root, dataset, splitBy)

    splits = ['train', 'val', 'test']
    refer.IMAGE_DIR = '/home/claytonfields/nlp/code/data/coco/images/mscoco/train2014'

    config = copy.deepcopy(_config)
    pl.seed_everything(config["seed"])

    model = METERTransformerSS(config)


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
            


    for epoch in range(epochs):
        losses, loss = train(model, train_ds, optimizer, loss_fn)
        print(f'Epoch: {epoch}, Loss:  {loss.item()}')  
        gold = evaluate(model, eval_ds)
        acc = np.average(gold)
        print(f'acurracy on test set {acc}')




if __name__ == "__main__":
    main()






































