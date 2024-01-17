import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np

from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings, BertModel, BertEncoder, BertLayer
from .bert_model import BertCrossLayer, BertAttention
from . import swin_transformer as swin
from . import heads, objectives, meter_utils
import meter.modules.vit_model as vit
from .clip_model import build_model, adapt_position_encoding
from .swin_helpers import swin_adapt_position_encoding
from transformers import RobertaConfig, RobertaModel
from transformers import ElectraConfig, ElectraModel
from transformers import AutoConfig, AutoModel

class METERTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        
        # ===================== Architecture ===================== #
        self.is_clip= ('ViT' in config['image_encoder']) # used on 86, 183, 317
        self.is_deit = ('deit' in config['image_encoder']) # used on 317
        # self.is_swin = ('swin' in config['image_encoder']) # not used 
        self.is_electra = ('electra' in config['text_encoder']) # used on 283
        
        # self.is_huggingface = config['hugging_face']
        
        self.fine_tune = (self.hparams.config["load_path"] != ""
            and not self.hparams.config["test_only"])
        self.test_only = (self.hparams.config["load_path"] != "" 
            and self.hparams.config["test_only"])

        # Intialize Text Encoder
        # if 'roberta' in config['text_encoder']:
        #     bert_config = RobertaConfig(
        #         vocab_size=config["vocab_size"],
        #         hidden_size=config["cross_layer_hidden_size"],
        #         # num_hidden_layers=config["num_layers"],
        #         num_attention_heads=config["num_cross_layer_heads"],
        #         intermediate_size=config["cross_layer_hidden_size"] * config["cross_layer_mlp_ratio"],
        #         max_position_embeddings=config["max_text_len"],
        #         hidden_dropout_prob=config["cross_layer_drop_rate"],
        #         attention_probs_dropout_prob=config["cross_layer_drop_rate"],
        #     )
        # elif 'electra' in config['text_encoder']:
        #     bert_config = ElectraConfig(
                
        #         vocab_size=config["vocab_size"],
        #         hidden_size=config["cross_layer_hidden_size"],
        #         # num_hidden_layers=config["num_layers"],
        #         num_attention_heads=config["num_cross_layer_heads"],
        #         intermediate_size=config["cross_layer_hidden_size"] * config["cross_layer_mlp_ratio"],
        #         max_position_embeddings=config["max_text_len"],
        #         hidden_dropout_prob=config["cross_layer_drop_rate"],
        #         attention_probs_dropout_prob=config["cross_layer_drop_rate"],
        #         )
        # else:
        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["cross_layer_hidden_size"],
            # num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_cross_layer_heads"],
            intermediate_size=config["cross_layer_hidden_size"] * config["cross_layer_mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["cross_layer_drop_rate"],
            attention_probs_dropout_prob=config["cross_layer_drop_rate"],
        )

        resolution_after=config['image_size']
        
        # Intialize Transform Laayers
        self.cross_modal_text_transform = nn.Linear(config['text_encoder_hidden_size'], config['cross_layer_hidden_size'])
        self.cross_modal_text_transform.apply(objectives.init_weights)
        self.cross_modal_image_transform = nn.Linear(config['image_encoder_hidden_size'], config['cross_layer_hidden_size'])
        self.cross_modal_image_transform.apply(objectives.init_weights)

        # Initialize Token Type Embeddings
        self.token_type_embeddings = nn.Embedding(2, config["cross_layer_hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        # Handle Distributed Case
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                if self.is_clip:
                    build_model(config['image_encoder'], resolution_after=resolution_after)
                else:
                    getattr(swin, self.hparams.config["image_encoder"])(
                        pretrained=True, config=self.hparams.config,
                    )

                if 'roberta' in config['text_encoder']:
                    RobertaModel.from_pretrained(config['text_encoder'])
                elif 'electra' in config['text_encoder']:
                    ElectraModel.from_pretrained(config['text_encoder'])
                else:
                    BertModel.from_pretrained(config['text_encoder'])

            torch.distributed.barrier()
            
        # Initialize Vision Encoder
        # if self.is_huggingface:
            # if self.fine_tune or self.test_only:
            #     visual_config = AutoConfig.from_pretrained(config['image_encoder'])
            #     self.image_encoder = AutoModel.from_config(visual_config)
            # else:
            #     
        self.image_encoder = AutoModel.from_pretrained(config['image_encoder'])
            
        # else:
        #     if self.is_clip:
        #         self.image_encoder = build_model(config['image_encoder'], resolution_after=resolution_after)
        #     elif self.is_deit:
        #         self.image_encoder = getattr(vit, self.hparams.config["image_encoder"])(
        #             pretrained=True, config=self.hparams.config
        #         )
        #     else:
        #         self.image_encoder = getattr(swin, self.hparams.config["image_encoder"])(
        #             pretrained=True, config=self.hparams.config,
        #         )
        #         self.avgpool = nn.AdaptiveAvgPool1d(1)
            
        # Freeze Parameters for self.image_encoder
        if config['freeze_image_encoder']:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        
        # Initialize text_encoder
        if 'roberta' in config['text_encoder']:
            self.text_transformer = RobertaModel.from_pretrained(config['text_encoder'])
        elif 'electra' in config['text_encoder']:
            self.text_transformer = ElectraModel.from_pretrained(config['text_encoder'])
        else:
            self.text_transformer = BertModel.from_pretrained(config['text_encoder'])
            
        # Freeze Parameters for self.text_transformer
        if config['freeze_text_encoder']:
            for param in self.text_transformer.parameters():
                param.requires_grad = False

        # Define Cross Modal Layers
        self.cross_modal_image_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(config['num_cross_layers'])])
        self.cross_modal_image_layers.apply(objectives.init_weights)
        self.cross_modal_text_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(config['num_cross_layers'])])
        self.cross_modal_text_layers.apply(objectives.init_weights)

        self.cross_modal_image_pooler = heads.Pooler(config["cross_layer_hidden_size"])
        self.cross_modal_image_pooler.apply(objectives.init_weights)
        self.cross_modal_text_pooler = heads.Pooler(config["cross_layer_hidden_size"])
        self.cross_modal_text_pooler.apply(objectives.init_weights)
        
        # ===================== Pretraining ===================== #
        # Masked Language Modeling
        if config["loss_names"]["mlm"] > 0:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)
        
        # Image Text Matching
        if config["loss_names"]["itm"] > 0:
            self.itm_score = heads.ITMHead(config["cross_layer_hidden_size"]*2)
            self.itm_score.apply(objectives.init_weights)

        hs = self.hparams.config["cross_layer_hidden_size"]

        # ===================== Downstream ===================== #
        # Initialize Visual Question Answering V2 Classifier
        if self.hparams.config["loss_names"]["vqa"] > 0:
            vs = self.hparams.config["vqav2_label_size"]
            self.vqa_classifier = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, vs),
            )
            self.vqa_classifier.apply(objectives.init_weights)

        # Load Previously Trained Modules
        # TODO: Make this method similar to test when ready to test
        if self.fine_tune:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            # if self.is_clip:
            #     state_dict = adapt_position_encoding(state_dict, after=resolution_after, patch_size=self.hparams.config['patch_size'])
            # else:
            #     state_dict = swin_adapt_position_encoding(state_dict, after=resolution_after, before=config['resolution_before'])
            self.load_state_dict(state_dict, strict=False)

        # Initialize NLVR2 Classifier
        if self.hparams.config["loss_names"]["nlvr2"] > 0:
            self.nlvr2_classifier = nn.Sequential(
                nn.Linear(hs * 4, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 2),
            )
            self.nlvr2_classifier.apply(objectives.init_weights)
            emb_data = self.token_type_embeddings.weight.data
            self.token_type_embeddings = nn.Embedding(3, hs)
            self.token_type_embeddings.apply(objectives.init_weights)
            self.token_type_embeddings.weight.data[0, :] = emb_data[0, :]
            self.token_type_embeddings.weight.data[1, :] = emb_data[1, :]
            self.token_type_embeddings.weight.data[2, :] = emb_data[1, :]

        # Initialize SNLI-VE Classifier
        if self.hparams.config["loss_names"]["snli"] > 0:
            self.snli_classifier = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 3),
            )
            self.snli_classifier.apply(objectives.init_weights)

        # Initialize Image-Text Recall Classifier
        if self.hparams.config["loss_names"]["irtr"] > 0:
            self.rank_output = nn.Linear(hs, 1)
            self.rank_output.weight.data = self.itm_score.fc.weight.data[1:, :]
            self.rank_output.bias.data = self.itm_score.fc.bias.data[1:]
            self.margin = 0.2
            for p in self.itm_score.parameters():
                p.requires_grad = False
        
        # Initialize Reference Resolution Classifier
        if self.hparams.config["loss_names"]['ref'] > 0:
            self.ref_classifier = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 1),
            )
            self.ref_classifier.apply(objectives.init_weights)

        meter_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================

        # if self.test_only:
        #     ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
        #     state_dict = ckpt["state_dict"]
        #     if self.is_clip:
        #         state_dict = adapt_position_encoding(state_dict, after=resolution_after, patch_size=self.hparams.config['patch_size'])
        #     else:
        #         state_dict = swin_adapt_position_encoding(state_dict, after=resolution_after, before=config['resolution_before'])
        #     self.load_state_dict(state_dict, strict=False)
        if self.test_only:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            # if self.is_clip:
            #     state_dict = adapt_position_encoding(state_dict, after=resolution_after, patch_size=self.hparams.config['patch_size'])
            # else:
            #     state_dict = swin_adapt_position_encoding(state_dict, after=resolution_after, before=config['resolution_before'])
            self.load_state_dict(state_dict, strict=False)

    def infer(
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
        text_masks = batch[f"text_masks"]

        text_embeds = self.text_transformer.embeddings(input_ids=text_ids)
        device = text_embeds.device
        input_shape = text_masks.size()
        extend_text_masks = self.text_transformer.get_extended_attention_mask(text_masks, input_shape, device)
        
        # Project Embeddings if Necessary
        if self.is_electra:
            if self.text_transformer.config.embedding_size != self.text_transformer.config.hidden_size:
                text_embeds = self.text_transformer.embeddings_project(text_embeds)
        
        # Process Text Embeddings
        for layer in self.text_transformer.encoder.layer:
            text_embeds = layer(text_embeds, extend_text_masks)[0]
        text_embeds = self.cross_modal_text_transform(text_embeds)
        
        # Process Image Input to Image Embeddings
        image_embeds = self.image_encoder(img)
        # if self.is_huggingface:
        image_embeds = image_embeds.last_hidden_state
        image_embeds = self.cross_modal_image_transform(image_embeds)
        image_masks = torch.ones((image_embeds.size(0), image_embeds.size(1)), dtype=torch.long, device=device)
        extend_image_masks = self.text_transformer.get_extended_attention_mask(image_masks, image_masks.size(), device)

        # Cross-Modal Processing
        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            ),
        )

        x, y = text_embeds, image_embeds
        for text_layer, image_layer in zip(self.cross_modal_text_layers, self.cross_modal_image_layers):
            x1 = text_layer(x, y, extend_text_masks, extend_image_masks)
            y1 = image_layer(y, x, extend_image_masks, extend_text_masks)
            x, y = x1[0], y1[0]

        text_feats, image_feats = x, y
        cls_feats_text = self.cross_modal_text_pooler(x)
        if self.is_clip or self.is_deit:
            cls_feats_image = self.cross_modal_image_pooler(y)
        else:
            avg_image_feats = self.avgpool(image_feats.transpose(1, 2)).view(image_feats.size(0), 1, -1)
            cls_feats_image = self.cross_modal_image_pooler(avg_image_feats)
        cls_feats = torch.cat([cls_feats_text, cls_feats_image], dim=-1)


        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
        }
        return ret

    def forward(self, batch):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        # Image Text Matching
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm(self, batch))

        # Visual Question Answering
        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch))

        # Natural Language for Visual Reasoning 2
        if "nlvr2" in self.current_tasks:
            ret.update(objectives.compute_nlvr2(self, batch))

        # SNLI Visual Entailment
        if "snli" in self.current_tasks:
            ret.update(objectives.compute_snli(self, batch))

        # Image Retrieval and Text Retrieval
        if "irtr" in self.current_tasks:
            ret.update(objectives.compute_irtr(self, batch))
            
        # Reference Resolution Task
        if 'ref' in self.current_tasks:
            ret.update(objectives.compute_ref(self, batch))
             
        return ret

    def training_step(self, batch, batch_idx):
        meter_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def on_train_epoch_end(self):
        meter_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        meter_utils.set_task(self)
        output = self(batch)

    def on_validation_epoch_end(self):
        meter_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        meter_utils.set_task(self)
        output = self(batch)
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))

        return ret

    def on_test_epoch_end(self):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        # if self.hparams.config["loss_names"]["vqa"] > 0:
        #     objectives.vqa_test_wrapup(outs, model_name)
        meter_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return meter_utils.set_schedule(self)
