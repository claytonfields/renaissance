#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 17:19:52 2024

@author: claytonfields
"""

import torch
import torch.nn as nn

from transformers.models.auto import AutoModel
from transformers.models.lxmert.modeling_lxmert import LxmertXLayer
from transformers.models.lxmert.configuration_lxmert import LxmertConfig

from renaissance.modeling.heads import Pooler, init_weights


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
        
        # Imported lazily to avoid an import cycle: the backbones package
        # __init__ pulls in the wrapper that imports this module.
        from renaissance.modeling.backbones.hf_loader import (
            hf_model_hidden_size,
            load_hf_encoder,
        )

        def _overrides(prefix):
            return {
                "hidden_size": config[f"{prefix}_hidden_size"],
                "num_hidden_layers": config[f"{prefix}_num_layers"],
                "num_attention_heads": config[f"{prefix}_num_heads"],
                "intermediate_size": config[f"{prefix}_hidden_size"]
                * config[f"{prefix}_mlp_ratio"],
                "hidden_dropout_prob": config[f"{prefix}_drop_rate"],
                "attention_probs_dropout_prob": config[f"{prefix}_drop_rate"],
            }

        attn_impl = "flash_attention_2" if config.get("use_flash_attention", False) else None

        # Vision encoder (built first — preserves the legacy RNG draw order
        # so fixed-seed init is byte-identical to the pre-refactor path).
        self.image_encoder, self.image_encoder_hidden_size = load_hf_encoder(
            config["image_encoder"],
            random_init=self.random_init_vision_encoder,
            overrides=(
                _overrides("image_encoder")
                if self.random_init_vision_encoder
                and config["image_encoder_manual_configuration"]
                else None
            ),
            freeze=config["freeze_image_encoder"],
            attn_implementation=attn_impl,
        )

        # Text encoder.
        self.text_transformer, _ = load_hf_encoder(
            config["text_encoder"],
            random_init=self.random_init_text_encoder,
            overrides=(
                _overrides("text_encoder")
                if self.random_init_text_encoder
                and config["text_encoder_manual_configuration"]
                else None
            ),
            freeze=config["freeze_text_encoder"],
            attn_implementation=attn_impl,
        )

        # Optional gradient checkpointing on the tower encoders. HF sets a
        # `.gradient_checkpointing` attribute on inner Transformer stacks that
        # each layer's forward reads at runtime. The LXMERT fusion is a
        # separate custom nn.Module — no HF hook, so it's not checkpointed.
        if config.get("gradient_checkpointing", False):
            self.image_encoder.gradient_checkpointing_enable()
            self.text_transformer.gradient_checkpointing_enable()

        self.text_transformer_hidden_size = hf_model_hidden_size(self.text_transformer)
        self.hidden_size = config['cross_layer_hidden_size']
        # Cross Modal Layers
        self.cross_modal_text_transform = nn.Linear(self.text_transformer_hidden_size, self.hidden_size)
        self.cross_modal_text_transform.apply(init_weights)
        self.cross_modal_image_transform = nn.Linear(self.image_encoder_hidden_size, self.hidden_size)
        self.cross_modal_image_transform.apply(init_weights)
        
        # Cross-Modal Module with LXMERT Layers
        # self.fusion_encoder = BertCrossModalEncoder(config)
        self.fusion_encoder = LxmertCrossModalEncoder(config)
        self.fusion_encoder.apply(init_weights)
        
        if config['freeze_cross_modal_layers']:
            for param in self.fusion_encoder.parameters():
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
        
        # text_embeds = self.text_transformer(inputs_embeds=text_embeds).last_hidden_state
        text_embeds = self.text_transformer(input_ids=text_ids).last_hidden_state
        text_embeds = self.cross_modal_text_transform(text_embeds)
        
        # Process Image Input to Image Embeddings
        if self.fine_tune or self.test_only:
            try:
                image_output = self.image_encoder(img, interpolate_pos_encoding = True)
            except:
                image_output = self.image_encoder(img)
        else:
            image_output = self.image_encoder(img)
            
        # if self.is_huggingface:
        image_embeds = image_output.last_hidden_state
        # image_dims = len(image_embeds.shape)
        if len(image_embeds.shape) == 4:
            image_embeds = self.resize_convolutional_output(image_output)
        
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
        
        # cls_feats, text_feats, image_feats = self.fusion_encoder(text_embeds, image_embeds, extend_text_masks, extend_image_masks)
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
    
    def resize_convolutional_output(self, image_output):
    
        # pooled_output = image_output.pooler_output
        # cls = pooled_output.squeeze(-1).permute((0,2,1))
        # cls_feature = image_output.pooler_output
        try:
            cls_feature = image_output.pooler_output
        except:
            try:
                cls_feature = image_output.cls_token_value
            except:
                cls_feature = image_output.last_hidden_state.flatten(2).mean(-1)
        cls_dims = len(cls_feature.shape)
        if cls_dims ==4:
            cls_feature = cls_feature.squeeze(-1).permute((0,2,1))
        elif cls_dims == 2:
            cls_feature = cls_feature.unsqueeze(1)
        # pooled_output = image_encoder.pooler(image_embeds)
        hidden_state = image_output.last_hidden_state
        hidden_state = hidden_state.flatten(start_dim=2, end_dim=3).permute((0,2,1))
        
        final_output = torch.concat((cls_feature, hidden_state), dim=1)
        return final_output
    
    def adjust_type_embeds_for_nlvr2(self):
        emb_data = self.token_type_embeddings.weight.data
        self.token_type_embeddings = nn.Embedding(3, self.hidden_size)
        self.token_type_embeddings.apply(init_weights)
        self.token_type_embeddings.weight.data[0, :] = emb_data[0, :]
        self.token_type_embeddings.weight.data[1, :] = emb_data[1, :]
        self.token_type_embeddings.weight.data[2, :] = emb_data[1, :]