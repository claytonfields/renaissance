#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 17:19:52 2024

@author: claytonfields
"""

import math
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F

# from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.auto import AutoConfig, AutoModel
from transformers.models.lxmert.modeling_lxmert import LxmertXLayer
from transformers.models.lxmert.configuration_lxmert import LxmertConfig

# from transformers.models.bert.modeling_bert import BertPredictionHeadTransform

from typing import List, Optional, Tuple, Union

from .objectives import init_weights
from .heads import Pooler

class LxmertCrossModalEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        lxmert_config = LxmertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["cross_layer_hidden_size"],
            num_attention_heads=config["num_cross_layer_heads"],
            intermediate_size=config["cross_layer_hidden_size"] * config["cross_layer_mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["cross_layer_drop_rate"],
            attention_probs_dropout_prob=config["cross_layer_drop_rate"],
        )        

        self.cross_modal_layers = nn.ModuleList([LxmertXLayer(lxmert_config) for _ in range(config['num_cross_layers'])])
        
        self.cross_modal_image_pooler = Pooler(config["cross_layer_hidden_size"])
        self.cross_modal_text_pooler = Pooler(config["cross_layer_hidden_size"])
        
    def forward(
        self,
        lang_feats,
        lang_attention_mask,
        visual_feats,
        visual_attention_mask,                  
        output_attentions=False,
    ):
        for layer in self.cross_modal_layers:
    
            x_outputs = layer(
                lang_feats,
                lang_attention_mask,
                visual_feats,
                visual_attention_mask   
            )
            lang_feats, visual_feats = x_outputs[:2]
            
        ### TODO!!! Add pooler to extract cls features
        cls_feats_text = self.cross_modal_text_pooler(lang_feats)
        cls_feats_image = self.cross_modal_image_pooler(visual_feats)
        cls_feats = torch.cat([cls_feats_text, cls_feats_image], dim=-1)
        
        return cls_feats, lang_feats, visual_feats
        


class TwoTowerEncoder(nn.Module):
    def __init__(
            self, 
            config,
            fine_tune,
            test_only
    ):
        super().__init__()
        
        self.fine_tune = fine_tune
        self.test_only = test_only
        
        self.random_init_vision_encoder = config['random_init_vision_encoder']
        self.random_init_text_encoder = config['random_init_text_encoder']
        
        # Vision Encoder
        if self.random_init_vision_encoder:
            if config['image_encoder_manual_configuration']:
                image_encoder_kwargs = {
                    'hidden_size' : config["image_encoder_hidden_size"],
                    'num_hidden_layers' : config["image_encoder_num_layers"],
                    'num_attention_heads' : config["image_encoder_num_heads"],
                    'intermediate_size' : config["image_encoder_hidden_size"] * config["image_encoder_mlp_ratio"],
                    'hidden_dropout_prob' : config["image_encoder_drop_rate"],
                    'attention_probs_dropout_prob' : config["image_encoder_drop_rate"],
                }
                hf_image_config = AutoConfig.from_pretrained(config['image_encoder'], **image_encoder_kwargs)
            # elif not config['image_encoder_manual_configuration']:
            else:
                hf_image_config = AutoConfig.from_pretrained(config['image_encoder'])
            self.image_encoder = AutoModel.from_config(hf_image_config)
            # if 'clip' in (config['image_encoder']):
            #     self.image_encoder = self.image_encoder.vision_model
        
        else:
            # hf_image_config = AutoConfig.from_pretrained(config['image_encoder'])
            self.image_encoder = AutoModel.from_pretrained(config['image_encoder'])
            
        # Freeze Parameters for self.image_encoder
        if config['freeze_image_encoder']:
            for param in self.image_encoder.parameters(self):
                param.requires_grad = False
        
        # Initialize text_encoder
        # Randomly Initialize Encoder Weights
        if self.random_init_text_encoder:
            if config['text_encoder_manual_configuration']:
                text_encoder_kwargs = {
                    'hidden_size' : config["text_encoder_hidden_size"],
                    'num_hidden_layers' : config["text_encoder_num_layers"],
                    'num_attention_heads' : config["text_encoder_num_heads"],
                    'intermediate_size' : config["text_encoder_hidden_size"] * config["text_encoder_mlp_ratio"],
                    'hidden_dropout_prob' : config["text_encoder_drop_rate"],
                    'attention_probs_dropout_prob' : config["text_encoder_drop_rate"],
                }
                hf_text_config = AutoConfig.from_pretrained(config['text_encoder'], **text_encoder_kwargs)
            # elif not config['text_encoder_manual_configuration']:
            else:
                hf_text_config = AutoConfig.from_pretrained(config['text_encoder'])
            self.text_transformer = AutoModel.from_config(hf_text_config)
        else:
            # hf_text_config = AutoConfig.from_pretrained(config['text_encoder'])
            self.text_transformer = AutoModel.from_pretrained(config['text_encoder'])
        
        # Freeze Parameters for self.text_transformer
        if config['freeze_text_encoder']:
            for param in self.text_transformer.parameters():
                param.requires_grad = False
        
        
        self.image_encoder_hidden_size = self.image_encoder.config.hidden_size
        self.text_transformer_hidden_size = self.text_transformer.config.hidden_size
        self.hidden_size = config['cross_layer_hidden_size']
        # Cross Modal Layers
        self.cross_modal_text_transform = nn.Linear(self.text_transformer_hidden_size, self.hidden_size)
        self.cross_modal_text_transform.apply(init_weights)
        self.cross_modal_image_transform = nn.Linear(self.image_encoder_hidden_size, self.hidden_size)
        self.cross_modal_image_transform.apply(init_weights)
        
        # Cross-Modal Module with LXMERT Layers
        self.fusion_encoder = LxmertCrossModalEncoder(config)
        self.fusion_encoder.apply(init_weights)
        
        if config['freeze_cross_modal_layers']:
            for param in self.fusion_encoder.parameters(self):
                param.requires_grad = False
        
        # Token Type Embeddings
        self.token_type_embeddings = nn.Embedding(2, config["cross_layer_hidden_size"])
        self.token_type_embeddings.apply(init_weights)
        

        # Handle Distributed Case
        # Test this on frege when time permits
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                AutoModel.from_pretrained(config['image_encoder'])
                AutoModel.from_pretrained(config['text_encoder'])
            torch.distributed.barrier()
            
    def forward(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        img=None,
    ):
        if img is None:
            if f"image_{image_token_type_idx - 1}" in batch:
                imgkey = f"image_{image_token_type_idx - 1}"
            else:
                imgkey = "image"
            img = batch[imgkey][0]
        
        # Process Text Input to Text Embeddings
        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch["text_masks"]

        text_embeds = self.text_transformer.embeddings(input_ids=text_ids)
        device = text_embeds.device
        input_shape = text_masks.size()
        extend_text_masks = self.text_transformer.get_extended_attention_mask(text_masks, input_shape)#, device)
        
        text_embeds = self.text_transformer(inputs_embeds=text_embeds).last_hidden_state
        text_embeds = self.cross_modal_text_transform(text_embeds)
        
        # Process Image Input to Image Embeddings
        if self.fine_tune or self.test_only:
            try:
                image_embeds = self.image_encoder(img, interpolate_pos_encoding = True)
            except:
                image_embeds = self.image_encoder(img)
        else:
            image_embeds = self.image_encoder(img)
            
        # if self.is_huggingface:
        image_embeds = image_embeds.last_hidden_state
        image_embeds = self.cross_modal_image_transform(image_embeds)
        image_masks = torch.ones((image_embeds.size(0), image_embeds.size(1)), dtype=torch.long, device=device)
        extend_image_masks = self.text_transformer.get_extended_attention_mask(image_masks, image_masks.size())#, device)

        # Cross-Modal Processing
        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            ),
        )
        
        cls_feats, text_feats, image_feats = self.fusion_encoder(text_embeds, extend_text_masks, image_embeds, extend_image_masks)

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
        }
        return ret
        
    def get_hidden_size(self):
        return self.hidden_size