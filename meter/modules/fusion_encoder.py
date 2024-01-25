#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 10:52:44 2024

@author: claytonfields
"""

import torch
import torch.nn as nn

from transformers.models.bert.modeling_bert import BertConfig, BertModel
from .bert_model import BertCrossLayer
from . import heads, objectives, meter_utils






class CrossModalEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Cross Modal Layers
        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["cross_layer_hidden_size"],
            num_attention_heads=config["num_cross_layer_heads"],
            intermediate_size=config["cross_layer_hidden_size"] * config["cross_layer_mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["cross_layer_drop_rate"],
            attention_probs_dropout_prob=config["cross_layer_drop_rate"],
        )
        # resolution_after=config['image_size']

        # self.cross_modal_text_transform = nn.Linear(
        #     config['text_encoder_hidden_size'], 
        #     config['cross_layer_hidden_size']
        #     #requires_grad = config['freeze_cross_modal_layers']
        # )
        # self.cross_modal_text_transform.apply(objectives.init_weights)
        # self.cross_modal_image_transform = nn.Linear(config['image_encoder_hidden_size'], config['cross_layer_hidden_size'])
        # self.cross_modal_image_transform.apply(objectives.init_weights)

        self.cross_modal_image_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(config['num_cross_layers'])])
        self.cross_modal_image_layers.apply(objectives.init_weights)
        self.cross_modal_text_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(config['num_cross_layers'])])
        self.cross_modal_text_layers.apply(objectives.init_weights)

        self.cross_modal_image_pooler = heads.Pooler(config["cross_layer_hidden_size"])
        self.cross_modal_image_pooler.apply(objectives.init_weights)
        self.cross_modal_text_pooler = heads.Pooler(config["cross_layer_hidden_size"])
        self.cross_modal_text_pooler.apply(objectives.init_weights)
        
    def forward(self, text_embeds, image_embeds, extend_text_masks, extend_image_masks):
        x, y = text_embeds, image_embeds
        for text_layer, image_layer in zip(self.cross_modal_text_layers, self.cross_modal_image_layers):
            x1 = text_layer(x, y, extend_text_masks, extend_image_masks)
            y1 = image_layer(y, x, extend_image_masks, extend_text_masks)
            x, y = x1[0], y1[0]

        text_feats, image_feats = x, y
        cls_feats_text = self.cross_modal_text_pooler(x)
        cls_feats_image = self.cross_modal_image_pooler(y)
        # original swin case
        # avg_image_feats = self.avgpool(image_feats.transpose(1, 2)).view(image_feats.size(0), 1, -1)
        # cls_feats_image = self.cross_modal_image_pooler(avg_image_feats)
        cls_feats = torch.cat([cls_feats_text, cls_feats_image], dim=-1)
        
        return cls_feats, text_feats, image_feats









