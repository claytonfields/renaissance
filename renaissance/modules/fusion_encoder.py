"""
Fusion encoder modules using cross-attention for two-tower vision language 
encoder models. Fusion encoder modules are used in 
renaissance/modules/renaissance_module.py.
"""

import torch
import torch.nn as nn

from transformers.models.bert.modeling_bert import BertConfig
from transformers.models.lxmert.modeling_lxmert import LxmertXLayer
from transformers.models.lxmert.configuration_lxmert import LxmertConfig
from transformers.models.auto import AutoConfig, AutoModel
from . import heads, objectives, renaissance_utils


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
        
class OneTowerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.random_init_encoder = config['random_init_encoder']
        if self.random_init_encoder:
            # Manually Configure Encoder Dimensions
            if config['encoder_manual_configuration']:
                encoder_kwargs = {
                    'vocab_size' : config["vocab_size"],
                    'hidden_size' : config["hidden_size"],
                    'num_hidden_layers' : config["num_layers"],
                    'num_attention_heads' : config["num_heads"],
                    'intermediate_size' : config["hidden_size"] * config["mlp_ratio"],
                    'max_position_embeddings' : config["max_text_len"],
                    'hidden_dropout_prob' : config["drop_rate"],
                    'attention_probs_dropout_prob' : config["drop_rate"],
                }
                hf_config = AutoConfig.from_pretrained(config['encoder'], **encoder_kwargs)
            # Use Default Encoder Dimensions with Random Weights
            elif not config['manual_configuration']:
                hf_config = AutoConfig.from_pretrained(config['encoder'])
            model = AutoModel.from_config(hf_config)
            self.encoder = model.encoder
            
            image_size = config['image_size']
            max_text_len = config['max_text_len']
            self.hidden_size = config['hidden_size']
            self.embedding_size = config['embedding_size']
        # Use Pretrained Encoder Weights from Huggingface Hub
        else:
            # Download Encoder - Get Dimensions
            model = AutoModel.from_pretrained(config['encoder'])
            self.encoder = model.encoder
            
    # Implement infer method for one_tower models
    def infer_one_tower(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"

        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
    
        text_embeds = self.text_embeddings(text_ids)
        
        image_embeds = self.image_embeddings(batch['image'][0], interpolate_pos_encoding=True)
        image_masks = torch.ones_like(image_embeds[:,:,0], dtype=torch.long)

        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                
                torch.full_like(image_masks, image_token_type_idx))
        )
        
        if self.embedding_size != self.hidden_size:
            text_embeds = self.text_embedding_projection(text_embeds)
            image_embeds = self.image_embedding_projection(image_embeds)
        
        # ERROR: Causes shape error with one-tower model.
        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        co_masks = torch.cat([text_masks, image_masks], dim=1)

        x = co_embeds

        # for i, blk in enumerate(self.encoder.blocks):
        #     x, _attn = blk(x, mask=co_masks)

        # x = self.transformer.norm(x)
        # try:
        #     x = self.encoder(inputs_embeds=x)[0]
        # except:
        x = self.encoder(x)[0]
        
        text_feats, image_feats = (
            x[:, : text_embeds.shape[1]],
            x[:, text_embeds.shape[1] :],
        )
        
        if self.pooler_type == 'single':
            cls_feats = self.pooler(x)
        else:
            cls_feats_text = self.text_pooler(text_feats)
            cls_feats_image = self.image_pooler(image_feats)
            cls_feats = torch.cat([cls_feats_text, cls_feats_image], dim=-1)
            

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            'text_labels' : text_labels,
            'text_ids' : text_ids
        }

        return ret