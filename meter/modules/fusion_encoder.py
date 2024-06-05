"""
Fusion encoder modules using cross-attention for two-tower vision language 
encoder models. Fusion encoder modules are used in 
meter/modules/meter_module.py.
"""

import torch
import torch.nn as nn

from transformers.models.bert.modeling_bert import BertConfig
from transformers.models.lxmert.modeling_lxmert import LxmertXLayer
from transformers.models.lxmert.configuration_lxmert import LxmertConfig
# from .bert_model import BertCrossLayer
from . import heads, objectives, meter_utils






# class BertCrossModalEncoder(nn.Module):
#     def __init__(self, config):
#         super().__init__()
        
        
#         bert_config = BertConfig(
#             vocab_size=config["vocab_size"],
#             hidden_size=config["cross_layer_hidden_size"],
#             num_attention_heads=config["num_cross_layer_heads"],
#             intermediate_size=config["cross_layer_hidden_size"] * config["cross_layer_mlp_ratio"],
#             max_position_embeddings=config["max_text_len"],
#             hidden_dropout_prob=config["cross_layer_drop_rate"],
#             attention_probs_dropout_prob=config["cross_layer_drop_rate"],
#         )

#         self.cross_modal_image_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(config['num_cross_layers'])])
#         self.cross_modal_text_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(config['num_cross_layers'])])

#         self.cross_modal_image_pooler = heads.Pooler(config["cross_layer_hidden_size"])
#         self.cross_modal_text_pooler = heads.Pooler(config["cross_layer_hidden_size"])
        
#     def forward(self, text_embeds, image_embeds, extend_text_masks, extend_image_masks):
#         x, y = text_embeds, image_embeds
#         for text_layer, image_layer in zip(self.cross_modal_text_layers, self.cross_modal_image_layers):
#             x1 = text_layer(x, y, extend_text_masks, extend_image_masks)
#             y1 = image_layer(y, x, extend_image_masks, extend_text_masks)
#             x, y = x1[0], y1[0]

#         text_feats, image_feats = x, y
#         cls_feats_text = self.cross_modal_text_pooler(x)
#         cls_feats_image = self.cross_modal_image_pooler(y)
#         cls_feats = torch.cat([cls_feats_text, cls_feats_image], dim=-1)
        
#         return cls_feats, text_feats, image_feats

class LxmertCrossModalEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        lxmert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["cross_layer_hidden_size"],
            num_attention_heads=config["num_cross_layer_heads"],
            intermediate_size=config["cross_layer_hidden_size"] * config["cross_layer_mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["cross_layer_drop_rate"],
            attention_probs_dropout_prob=config["cross_layer_drop_rate"],
        )        

        self.cross_modal_layers = nn.ModuleList([LxmertXLayer(lxmert_config) for _ in range(config['num_cross_layers'])])
        
        self.cross_modal_image_pooler = heads.Pooler(config["cross_layer_hidden_size"])
        self.cross_modal_text_pooler = heads.Pooler(config["cross_layer_hidden_size"])
        
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
        


