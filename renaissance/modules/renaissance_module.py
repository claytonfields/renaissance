import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from transformers.models.bert.modeling_bert import BertConfig#, BertModel, BertEmbeddings
from transformers.models.vit.modeling_vit import ViTEmbeddings, ViTConfig
from transformers.models.electra.modeling_electra import  ElectraConfig#,ElectraEmbeddings
from .embeddings import ElectraEmbeddings
# from transformers.model.vit import 
# from .bert_model import BertCrossLayer
from . import heads, objectives, renaissance_utils
from transformers import AutoConfig, AutoModel#, AutoModelForSequenceClassification
from .fusion_encoder import LxmertCrossModalEncoder

class RenaissanceTransformer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.model_type = config['model_type']
        
        # ===================== BaseArchitecture ===================== #
        # Adjust dimensions for fine-tuning
        self.fine_tune = (self.hparams.config["load_path"] != ""
            and not self.hparams.config["test_only"])
        self.test_only = (self.hparams.config["load_path"] != "" 
            and self.hparams.config["test_only"])
        
        
        if self.fine_tune or self.test_only:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            # UNCOMMENT BELOW WHEN DONE TESTING VQA!!!!!!!
            self.original_max_text_len = ckpt['hyper_parameters']['config']['max_text_len']
            self.new_max_text_len = config['max_text_len']
            self.original_image_size = ckpt['hyper_parameters']['config']['original_image_size']
            self.new_image_size = config['image_size']
            
        # else:
        #     self.hparams.config['original_max_text_len'] = config['max_text_len']
        #     # self.new_max_text_len = config['max_text_len']
        #     self.hparams.config['original_image_size'] = config['image_size']
        #     # self.new_image_size = config['image_size']
        
        # self.save_hyperparameters()
        
        if self.model_type == 'one-tower':
            
            # self.encoder_type = config['encoder_type']
            self.random_init_encoder = config['random_init_encoder']
            self.pooler_type = config['pooler_type']
            
            if self.random_init_encoder:
                hf_config = BertConfig(
                    vocab_size=config["vocab_size"],
                    hidden_size=config["hidden_size"],
                    num_hidden_layers=config["num_layers"],
                    num_attention_heads=config["num_heads"],
                    intermediate_size=config["hidden_size"] * config["mlp_ratio"],
                    max_position_embeddings=config["max_text_len"],
                    hidden_dropout_prob=config["drop_rate"],
                    attention_probs_dropout_prob=config["drop_rate"],
                )
                self.encoder = AutoModel.from_config(hf_config)
            else:
                # Download Encoder - Get Dimensions
                model = AutoModel.from_pretrained(config['encoder'])
                self.encoder = model.encoder
                self.hidden_size = self.encoder.config.hidden_size
                try:
                    self.embedding_size = self.encoder.config.embedding_size
                except:
                    self.embedding_size = self.hidden_size
                
                if self.embedding_size != self.hidden_size:
                    self.text_embedding_projection = nn.Linear(self.embedding_size, self.hidden_size)
                    self.image_embedding_projection = nn.Linear(self.embedding_size, self.hidden_size)
                
                if self.fine_tune or self.test_only:
                    image_size = self.original_image_size
                    max_text_len = self.original_max_text_len
                else:
                    image_size = config['image_size']
                    max_text_len = config['max_text_len']
            
                image_config = ViTConfig(
                    image_size=image_size,
                    patch_size=config['patch_size'],
                    hidden_size=self.embedding_size,
                    hidden_dropout_prob=config["drop_rate"],
                    attention_probs_dropout_prob=config["drop_rate"],
                )
                
                text_config = ElectraConfig(
                    vocab_size=config["vocab_size"],
                    hidden_size=self.hidden_size,
                    embedding_size=self.embedding_size,
                    max_position_embeddings=max_text_len,
                    hidden_dropout_prob=config["drop_rate"],
                    attention_probs_dropout_prob=config["drop_rate"],
                )
            
            # Add ability to adjust embedding size for down stream changes
            self.text_embeddings = ElectraEmbeddings(text_config)
            self.text_embeddings.apply(objectives.init_weights)
            
            self.image_embeddings = ViTEmbeddings(image_config)
            self.image_embeddings.apply(objectives.init_weights)
            
            self.token_type_embeddings = nn.Embedding(2, self.embedding_size)
            self.token_type_embeddings.apply(objectives.init_weights)
                
            if self.pooler_type == 'single':
                self.pooler = heads.Pooler(self.hidden_size)
                self.pooler.apply(objectives.init_weights)
            elif self.pooler_type =='double':
                self.text_pooler = heads.Pooler(self.hidden_size)
                self.text_pooler.apply(objectives.init_weights)
                self.image_pooler = heads.Pooler(self.hidden_size)
                self.image_pooler.apply(objectives.init_weights)
        
        elif self.model_type == 'two-tower':
            # ===================== BaseArchitecture ===================== #
            self.random_init_vision_encoder = config['random_init_vision_encoder']
            self.random_init_text_encoder = config['random_init_text_encoder']
            
            self.hidden_size = config['cross_layer_hidden_size']
            # Cross Modal Layers
            self.cross_modal_text_transform = nn.Linear(config['text_encoder_hidden_size'], config['cross_layer_hidden_size'])
            self.cross_modal_text_transform.apply(objectives.init_weights)
            self.cross_modal_image_transform = nn.Linear(config['image_encoder_hidden_size'], config['cross_layer_hidden_size'])
            self.cross_modal_image_transform.apply(objectives.init_weights)
            
            # Cross-Modal Module with LXMERT Layers
            self.fusion_encoder = LxmertCrossModalEncoder(config)
            self.fusion_encoder.apply(objectives.init_weights)
            
            
            
            if config['freeze_cross_modal_layers']:
                for param in self.fusion_encoder.parameters(self):
                    param.requires_grad = False
            
    
            # Token Type Embeddings
            self.token_type_embeddings = nn.Embedding(2, config["cross_layer_hidden_size"])
            self.token_type_embeddings.apply(objectives.init_weights)
            
    
            # Handle Distributed Case
            # Test this on frege when time permits
            if torch.distributed.is_initialized():
                if torch.distributed.get_rank() == 0:
                    AutoModel.from_pretrained(config['image_encoder'])
                    AutoModel.from_pretrained(config['text_encoder'])
                torch.distributed.barrier()
                
            # Vision Encoder
            if not self.random_init_vision_encoder:
                self.image_encoder = AutoModel.from_pretrained(config['image_encoder'])
                if 'clip' in (config['image_encoder']):
                    self.image_encoder = self.image_encoder.vision_model
                    
            else:
                visual_kwargs = None
                visual_config = AutoConfig.from_pretrained(config['image_encoder'], kwargs=visual_kwargs)
                self.image_encoder = AutoModel.from_config(visual_config)
                
            # Freeze Parameters for self.image_encoder
            if config['freeze_image_encoder']:
                for param in self.image_encoder.parameters(self):
                    param.requires_grad = False
            
            # Initialize text_encoder
            if not self.random_init_text_encoder:
                self.text_transformer = AutoModel.from_pretrained(config['text_encoder'])
            else:
                text_kwargs = None
                text_config = AutoConfig.from_pretrained(config['text_encoder'], kwargs=text_kwargs)
                self.text_transformer = AutoModel.from_config(text_config)
            
            # Freeze Parameters for self.text_transformer
            if config['freeze_text_encoder']:
                for param in self.text_transformer.parameters():
                    param.requires_grad = False
        else:
            raise TypeError('Model Type not supported.')
        
        # ===================== Pretraining ===================== #
        
        if self.model_type =='one-tower':
            if self.pooler_type == 'single':
                hs = self.hs
            elif self.pooler_type == 'double':
                hs = 2*self.hidden_size
        else:
            hs = 2*self.hparams.config["cross_layer_hidden_size"]
        
        # Masked Language Modeling
        if self.hparams.config["loss_names"]["mlm"] > 0:
            self.mlm_score = heads.MLMHead(config, hidden_size=self.hidden_size)
            self.mlm_score.apply(objectives.init_weights)
        
        # Image Text Matching
        if config["loss_names"]["itm"] > 0:
            self.itm_score = heads.ITMHead(hs)
            self.itm_score.apply(objectives.init_weights)

        
        # ===================== Downstream  ===================== #
        
        # Initialize Visual Question Answering V2 Classifier
        if self.hparams.config["loss_names"]["vqa"] > 0:
            vs = self.hparams.config["vqav2_label_size"]
            self.vqa_classifier = nn.Sequential(
                nn.Linear(hs, hs),
                nn.LayerNorm(hs ),
                nn.GELU(),
                nn.Linear(hs, vs),
            )
            self.vqa_classifier.apply(objectives.init_weights)

        # Load Previously Trained Modules
        if self.fine_tune:
            self.load_state_dict(state_dict, strict=False)
            if (self.model_type == 'one-tower') and (self.original_max_text_len != self.new_max_text_len):
                self.text_embeddings._adjust_position_embeddings(self.new_max_text_len)
            

        # Initialize NLVR2 Classifier
        # May cause error in two-tower model!
        if self.hparams.config["loss_names"]["nlvr2"] > 0:
            self.nlvr2_classifier = nn.Sequential(
                nn.Linear(hs * 2, hs),
                nn.LayerNorm(hs),
                nn.GELU(),
                nn.Linear(hs, 2),
            )
            self.nlvr2_classifier.apply(objectives.init_weights)
            emb_data = self.token_type_embeddings.weight.data
            # Possible error with wrong hidden size below
            self.token_type_embeddings = nn.Embedding(3, hs)
            self.token_type_embeddings.apply(objectives.init_weights)
            self.token_type_embeddings.weight.data[0, :] = emb_data[0, :]
            self.token_type_embeddings.weight.data[1, :] = emb_data[1, :]
            self.token_type_embeddings.weight.data[2, :] = emb_data[1, :]

        # Initialize SNLI-VE Classifier
        if self.hparams.config["loss_names"]["snli"] > 0:
            self.snli_classifier = nn.Sequential(
                nn.Linear(hs, hs),
                nn.LayerNorm(hs),
                nn.GELU(),
                nn.Linear(hs, 3),
            )
            self.snli_classifier.apply(objectives.init_weights)

        # Initialize Image-Text Recall Classifier
        # Possible error for two tower model below
        if self.hparams.config["loss_names"]["irtr"] > 0:
            self.rank_output = nn.Linear(self.cross_layer_hs, 1)
            self.rank_output.weight.data = self.itm_score.fc.weight.data[1:, :]
            self.rank_output.bias.data = self.itm_score.fc.bias.data[1:]
            self.margin = 0.2
            for p in self.itm_score.parameters():
                p.requires_grad = False
        
        # Initialize Reference Resolution Classifier
        if self.hparams.config["loss_names"]['ref'] > 0:
            self.ref_classifier = nn.Sequential(
                nn.Linear(hs, hs),
                nn.LayerNorm(hs),
                nn.GELU(),
                nn.Linear(hs, 1),
            )
            self.ref_classifier.apply(objectives.init_weights)
        
        # Text-Only Classification
        if self.model_type == 'one-tower':
            self.text_hs = self.hidden_size
        else:
            self.text_hs = config['text_encoder_hidden_size']
        
        
        self.text_only = False
     
        # MRPC Text Classifier
        if self.hparams.config["loss_names"]['mrpc'] > 0:
            # self.text_only = True
            # hidden_size = self.text_hs
            # num_labels = 2
            self.mrpc_classifier = heads.TextClassificationHead(
                hidden_size=self.text_hs, 
                num_labels=2
            )
            self.mrpc_classifier.apply(objectives.init_weights)
            
        
        # rte Text Classifier
        if self.hparams.config["loss_names"]['rte'] > 0:
            # self.text_only = True
            # hidden_size = sel
            # num_labels = 2
            self.rte_classifier = heads.TextClassificationHead(
                hidden_size=self.text_hs, 
                num_labels=2
            )
            self.rte_classifier.apply(objectives.init_weights)
        
        # wnli Text Classifier
        if self.hparams.config["loss_names"]['wnli'] > 0:
            # self.text_only = True
            self.wnli_classifier = self.rte_classifier = heads.TextClassificationHead(
                hidden_size=self.text_hs, 
                num_labels=2
            )
            self.wnli_classifier.apply(objectives.init_weights)
            
        # sst2 Text Classifier
        if self.hparams.config["loss_names"]['sst2'] > 0:
            # self.text_only = True
            self.sst2_classifier = self.rte_classifier = heads.TextClassificationHead(
                hidden_size=self.text_hs, 
                num_labels=2
            )
            self.sst2_classifier.apply(objectives.init_weights)
            
        # qqp Text Classifier
        if self.hparams.config["loss_names"]['qqp'] > 0:
            # self.text_only = True
            self.qqp_classifier = self.rte_classifier = heads.TextClassificationHead(
                hidden_size=self.text_hs, 
                num_labels=2
            )
            self.qqp_classifier.apply(objectives.init_weights)
            
        # qnli Text Classifier
        if self.hparams.config["loss_names"]['qnli'] > 0:
            # self.text_only = True
            self.qnli_classifier = self.rte_classifier = heads.TextClassificationHead(
                hidden_size=self.text_hs, 
                num_labels=2
            )
            self.qnli_classifier.apply(objectives.init_weights)
            
        # mnli Text Classifier
        if self.hparams.config["loss_names"]['mnli'] > 0:
            # self.text_only = True
            self.mnli_classifier = self.rte_classifier = heads.TextClassificationHead(
                hidden_size=self.text_hs, 
                num_labels=3
            )
            self.mnli_classifier.apply(objectives.init_weights)
        # cola Text Classifier
        if self.hparams.config["loss_names"]['cola'] > 0:
            # self.text_only = True
            self.cola_classifier = self.rte_classifier = heads.TextClassificationHead(
                hidden_size=self.text_hs, 
                num_labels=2
            )
            self.cola_classifier.apply(objectives.init_weights)
        
        # if self.text_only:
        #     self.text_classification_pooler = heads.Pooler(self.text_hs)
        #     self.text_classification_pooler.apply(objectives.init_weights)
            
        
        ### Image-Only Tasks ###
        # Image-Only Classfification
        self.image_only = False
        
        if self.model_type == 'one-tower':
            self.image_hs = self.hidden_size
        else:
            self.image_hs = config['image_encoder_hidden_size']
        
        # CIFAR-10 Image Classifier
        if self.hparams.config["loss_names"]['cifar10'] > 0:
            self.image_only = True
            self.cifar10_classifier = nn.Sequential(
                nn.Linear(self.image_hs, self.image_hs),
                nn.LayerNorm(self.image_hs),
                nn.GELU(),
                nn.Linear(self.image_hs, 10)
            )
            self.cifar10_classifier.apply(objectives.init_weights)
        
        if self.image_only:
            # Image-Only Classification Pooler
            self.image_classification_pooler = heads.Pooler(self.image_hs)
            self.image_classification_pooler.apply(objectives.init_weights)
            
        
        renaissance_utils.set_metrics(self)
        self.current_tasks = list()

        # Load Downstream (test_only = True)
        if self.test_only:
            # ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            # state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)
            
    def _freeze_cross_modal_layers(self):
        self._freeze_layer(self.cross_modal_text_transform)
        self._freeze_layer(self.cross_modal_image_transform)
        self._freeze_layer(self.cross_modal_image_layers)
        self._freeze_layer(self.cross_modal_text_layers)
        self._freeze_layer(self.cross_modal_image_pooler )
        self._freeze_layer(self.cross_modal_text_pooler)
        
    def _freeze_layer(self, layer):
        for param in layer.parameters():
            param.requires_grad = False
            # return self
            
    def infer(self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        img=None,
        image_embeds=None,
        image_masks=None,
    ):
        if self.model_type == 'one-tower':
            ret = self.infer_one_tower(
                batch,
                mask_text=mask_text,
                mask_image=mask_image,
                image_token_type_idx=image_token_type_idx,
                image_embeds=image_embeds,
                image_masks=image_masks,
            )
        elif self.model_type == 'two-tower':
            ret = self.infer_two_tower(
                batch,
                mask_text=mask_text,
                mask_image=mask_image,
                image_token_type_idx=image_token_type_idx,
                img=img
            )
        return ret
    
    def infer_two_tower(
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
        try:
            x = self.encoder(inputs_embeds=x)[0]
        except:
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
    
    # Review and if update, if needed for one-tower
    def infer_text_only(self, batch):
        hidden_state = self.text_transformer(**batch).last_hidden_state#.squeeze()
        # cls_feat
        # cls_feat = self.text_classification_pooler(hidden_state)
        
        return hidden_state

    # This is ugly. Try to generalize
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
        
        # Text Only Tasks
        
        # MRPC Task from GLUE
        if 'mrpc' in self.current_tasks:
            ret.update(objectives.compute_mrpc(self, batch))
        
        # rte Task from GLUE
        if 'rte' in self.current_tasks:
            ret.update(objectives.compute_rte(self, batch))
        
        # wnli Task from GLUE
        if 'wnli' in self.current_tasks:
            ret.update(objectives.compute_wnli(self, batch))
            
        # sst2 Task from GLUE
        if 'sst2' in self.current_tasks:
            ret.update(objectives.compute_sst2(self, batch))
            
        # qqp Task from GLUE
        if 'qqp' in self.current_tasks:
            ret.update(objectives.compute_qqp(self, batch))
            
        # qnli Task from GLUE
        if 'qnli' in self.current_tasks:
            ret.update(objectives.compute_qnli(self, batch))
            
        # mnli Task from GLUE
        if 'mnli' in self.current_tasks:
            ret.update(objectives.compute_mnli(self, batch))
            
        # cola Task from GLUE
        if 'cola' in self.current_tasks:
            ret.update(objectives.compute_cola(self, batch))
        
        # cifar10 Image-Only Classification Task
        if 'cifar10' in self.current_tasks:
            ret.update(objectives.compute_cifar10(self, batch))
            
        return ret

    def training_step(self, batch, batch_idx):
        renaissance_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def on_train_epoch_end(self):
        renaissance_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        renaissance_utils.set_task(self)
        output = self(batch)

    def on_validation_epoch_end(self):
        renaissance_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        renaissance_utils.set_task(self)
        output = self(batch)
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))

        return ret

    def on_test_epoch_end(self):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]
        # if self.hparams.config["loss_names"]["vqa"] > 0:
        #     objectives.vqa_test_wrapup(outs, model_name)
        renaissance_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return renaissance_utils.set_schedule(self)
