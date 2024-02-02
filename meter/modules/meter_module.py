import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from transformers.models.bert.modeling_bert import BertConfig, BertModel
from .bert_model import BertCrossLayer
from . import heads, objectives, meter_utils
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification
from .fusion_encoder import CrossModalEncoder

class METERTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        
        # ===================== BaseArchitecture ===================== #
        self.is_electra = ('electra' in config['text_encoder']) # used on 283
        self.fine_tune = (self.hparams.config["load_path"] != ""
            and not self.hparams.config["test_only"])
        self.test_only = (self.hparams.config["load_path"] != "" 
            and self.hparams.config["test_only"])

        self.random_init_vision_encoder = config['random_init_vision_encoder']
        self.random_init_text_encoder = config['random_init_text_encoder']

        # Handle Distributed Case
        # Test this on frege when time permits
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                AutoModel.from_pretrained(config['image_encoder'])
                AutoModel.from_pretrained(config['text_encoder'])
            torch.distributed.barrier()
            
        # Vision Encoder
        if self.random_init_vision_encoder:
            image_kwargs = None
            image_config = AutoConfig.from_pretrained(config['image_encoder'], kwargs=image_kwargs)
            self.image_encoder = AutoModel.from_config(image_config)
        else:
            # visual_config = AutoConfig.from_pretrained(config['image_encoder'])
            self.image_encoder = AutoModel.from_pretrained(config['image_encoder'])
        
        self.image_config = self.image_encoder.config
        # original swin case
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
            
        # Freeze Parameters for self.image_encoder
        if config['freeze_image_encoder']:
            for param in self.image_encoder.parameters(self):
                param.requires_grad = False
        
        # Initialize text_encoder
        if self.random_init_text_encoder:
            text_kwargs = None
            text_config = AutoConfig.from_pretrained(config['text_encoder'], kwargs=text_kwargs)
            self.text_transformer = AutoModel.from_config(text_config)
        else:
            self.text_transformer = AutoModel.from_pretrained(config['text_encoder'])
        
        self.text_config = self.text_transformer.config
        
        # Freeze Parameters for self.text_transformer
        if config['freeze_text_encoder']:
            for param in self.text_transformer.parameters():
                param.requires_grad = False
        
        # Dimensions
        self.image_hs = self.image_config.hidden_size
        
        self.text_hs = self.text_config.hidden_size
        self.vocab_size = self.text_config.vocab_size
        
        self.cross_layer_hs = config['cross_layer_hidden_size']
        self.num_cross_layer_heads = config['num_cross_layer_heads']
        self.cross_layer_mlp_ratio =  config['cross_layer_mlp_ratio']
        self.max_text_len = config['max_text_len']
        self.cross_layer_drop_rate = config['cross_layer_drop_rate']
        self.num_cross_layers = config['num_cross_layers']
        
        # Cross Modal Layers
        bert_config = BertConfig(
            vocab_size = self.vocab_size,
            hidden_size = self.cross_layer_hs,
            num_attention_heads = self.num_cross_layer_heads,
            intermediate_size = self.cross_layer_hs * self.cross_layer_mlp_ratio,
            max_position_embeddings = self.max_text_len,
            hidden_dropout_prob = self.cross_layer_drop_rate,
            attention_probs_dropout_prob = self.cross_layer_drop_rate,
        )
        # resolution_after=config['image_size']
        
        self.cross_modal_text_transform = nn.Linear(self.text_hs, self.cross_layer_hs)
        self.cross_modal_text_transform.apply(objectives.init_weights)
        self.cross_modal_image_transform = nn.Linear(self.image_hs, self.cross_layer_hs)
        self.cross_modal_image_transform.apply(objectives.init_weights)
        
        self.cross_modal_image_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(self.num_cross_layers)])
        self.cross_modal_image_layers.apply(objectives.init_weights)
        self.cross_modal_text_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(self.num_cross_layers)])
        self.cross_modal_text_layers.apply(objectives.init_weights)

        self.cross_modal_image_pooler = heads.Pooler(self.cross_layer_hs)
        self.cross_modal_image_pooler.apply(objectives.init_weights)
        self.cross_modal_text_pooler = heads.Pooler(self.cross_layer_hs)
        self.cross_modal_text_pooler.apply(objectives.init_weights)
        
        if config['freeze_cross_modal_layers']:
            self._freeze_cross_modal_layers()
        # self.fusion_encoder = CrossModalEncoder(config)
        # self.fusion_encoder.apply(objectives.init_weights)
        

        # Token Type Embeddings
        self.token_type_embeddings = nn.Embedding(2, self.cross_layer_hs)
        self.token_type_embeddings.apply(objectives.init_weights)
        
        # ===================== Pretraining ===================== #
        
        # Masked Language Modeling
        if self.hparams.config["loss_names"]["mlm"] > 0:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)
        
        # Image Text Matching
        if self.hparams.config["loss_names"]["itm"] > 0:
            self.itm_score = heads.ITMHead(self.cross_layer_hs*2)
            self.itm_score.apply(objectives.init_weights)

        # ===================== Downstream  ===================== #
        
        # Initialize Visual Question Answering V2 Classifier
        if self.hparams.config["loss_names"]["vqa"] > 0:
            vs = self.hparams.config["vqav2_label_size"]
            self.vqa_classifier = nn.Sequential(
                nn.Linear(self.cross_layer_hs * 2, self.cross_layer_hs * 2),
                nn.LayerNorm(self.cross_layer_hs * 2),
                nn.GELU(),
                nn.Linear(self.cross_layer_hs * 2, vs),
            )
            self.vqa_classifier.apply(objectives.init_weights)

        # Load Previously Trained Modules
        if self.fine_tune:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)

        # Initialize NLVR2 Classifier
        if self.hparams.config["loss_names"]["nlvr2"] > 0:
            self.nlvr2_classifier = nn.Sequential(
                nn.Linear(self.cross_layer_hs * 4, self.cross_layer_hs * 2),
                nn.LayerNorm(self.cross_layer_hs * 2),
                nn.GELU(),
                nn.Linear(self.cross_layer_hs * 2, 2),
            )
            self.nlvr2_classifier.apply(objectives.init_weights)
            emb_data = self.token_type_embeddings.weight.data
            self.token_type_embeddings = nn.Embedding(3, self.cross_layer_hs)
            self.token_type_embeddings.apply(objectives.init_weights)
            self.token_type_embeddings.weight.data[0, :] = emb_data[0, :]
            self.token_type_embeddings.weight.data[1, :] = emb_data[1, :]
            self.token_type_embeddings.weight.data[2, :] = emb_data[1, :]

        # Initialize SNLI-VE Classifier
        if self.hparams.config["loss_names"]["snli"] > 0:
            self.snli_classifier = nn.Sequential(
                nn.Linear(self.cross_layer_hs * 2, self.cross_layer_hs * 2),
                nn.LayerNorm(self.cross_layer_hs * 2),
                nn.GELU(),
                nn.Linear(self.cross_layer_hs * 2, 3),
            )
            self.snli_classifier.apply(objectives.init_weights)

        # Initialize Image-Text Recall Classifier
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
                nn.Linear(self.cross_layer_hs * 2, self.cross_layer_hs * 2),
                nn.LayerNorm(self.cross_layer_hs * 2),
                nn.GELU(),
                nn.Linear(self.cross_layer_hs * 2, 1),
            )
            self.ref_classifier.apply(objectives.init_weights)
        
        # Text-Only Classification
        # self.text_hs = config['text_encoder_hidden_size']
        self.text_classification_pooler = heads.Pooler(self.text_hs)
        self.text_classification_pooler.apply(objectives.init_weights)
        
        # MRPC Text Classifier
        if self.hparams.config["loss_names"]['mrpc'] > 0:
            self.mrpc_classifier = nn.Sequential(
                nn.Linear(self.text_hs, self.text_hs),
                nn.LayerNorm(self.text_hs),
                nn.GELU(),
                nn.Linear(self.text_hs, 2)
            )
            self.mrpc_classifier.apply(objectives.init_weights)
            # self.load_text_classifier()
        
        # rte Text Classifier
        if self.hparams.config["loss_names"]['rte'] > 0:
            self.rte_classifier = nn.Sequential(
                nn.Linear(self.text_hs, self.text_hs),
                nn.LayerNorm(self.text_hs),
                nn.GELU(),
                nn.Linear(self.text_hs, 2)
            )
            self.rte_classifier.apply(objectives.init_weights)
        
        # wnli Text Classifier
        if self.hparams.config["loss_names"]['wnli'] > 0:
            self.wnli_classifier = nn.Sequential(
                nn.Linear(self.text_hs, self.text_hs),
                nn.LayerNorm(self.text_hs),
                nn.GELU(),
                nn.Linear(self.text_hs, 2)
            )
            self.wnli_classifier.apply(objectives.init_weights)
            
        # sst2 Text Classifier
        if self.hparams.config["loss_names"]['sst2'] > 0:
            self.sst2_classifier = nn.Sequential(
                nn.Linear(self.text_hs, self.text_hs),
                nn.LayerNorm(self.text_hs),
                nn.GELU(),
                nn.Linear(self.text_hs, 2)
            )
            self.sst2_classifier.apply(objectives.init_weights)
            
        # qqp Text Classifier
        if self.hparams.config["loss_names"]['qqp'] > 0:
            self.qqp_classifier = nn.Sequential(
                nn.Linear(self.text_hs, self.text_hs),
                nn.LayerNorm(self.text_hs),
                nn.GELU(),
                nn.Linear(self.text_hs, 2)
            )
            self.qqp_classifier.apply(objectives.init_weights)
            
        # qnli Text Classifier
        if self.hparams.config["loss_names"]['qnli'] > 0:
            self.qnli_classifier = nn.Sequential(
                nn.Linear(self.text_hs, self.text_hs),
                nn.LayerNorm(self.text_hs),
                nn.GELU(),
                nn.Linear(self.text_hs, 2)
            )
            self.qnli_classifier.apply(objectives.init_weights)
            
        # mnli Text Classifier
        if self.hparams.config["loss_names"]['mnli'] > 0:
            self.mnli_classifier = nn.Sequential(
                nn.Linear(self.text_hs, self.text_hs),
                nn.LayerNorm(self.text_hs),
                nn.GELU(),
                nn.Linear(self.text_hs, 3)
            )
        
        # cola Text Classifier
        if self.hparams.config["loss_names"]['cola'] > 0:
            self.cola_classifier = nn.Sequential(
                nn.Linear(self.text_hs, self.text_hs),
                nn.LayerNorm(self.text_hs),
                nn.GELU(),
                nn.Linear(self.text_hs, 2)
            )
            self.cola_classifier.apply(objectives.init_weights)
            
        meter_utils.set_metrics(self)
        self.current_tasks = list()

        # Load Downstream (test_only = True)
        if self.test_only:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
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
        text_masks = batch["text_masks"]

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
        if self.fine_tune:
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
        cls_feats_image = self.cross_modal_image_pooler(y)
        cls_feats = torch.cat([cls_feats_text, cls_feats_image], dim=-1)
        # cls_feats, text_feats, image_feats = self.fusion_encoder(text_embeds, image_embeds, extend_text_masks, extend_image_masks)

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
    def infer_one_tower(self, batch):
        pass
    
    # Implement text only infer method
    def infer_text_only(self, batch):
        hidden_state = self.text_transformer(**batch).last_hidden_state#.squeeze()
        cls_feat = self.text_classification_pooler(hidden_state)
        
        return cls_feat
    
    def load_text_classifier(self):
        # self.text_encoder.save_pretrained('temp')
        self.text_encoder = AutoModelForSequenceClassification.from_pretrained('google/electra-small-discriminator')
     

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
