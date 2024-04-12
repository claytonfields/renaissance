import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertPredictionHeadTransform


class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ITMHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.fc(x)
        return x


class MLMHead(nn.Module):
    def __init__(self, config, weight=None):
        
        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["cross_layer_hidden_size"],
            num_attention_heads=config["num_cross_layer_heads"],
            intermediate_size=config["cross_layer_hidden_size"] * config["cross_layer_mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["cross_layer_drop_rate"],
            attention_probs_dropout_prob=config["cross_layer_drop_rate"],
        )
        super().__init__()
        self.transform = BertPredictionHeadTransform(bert_config)
        self.decoder = nn.Linear(bert_config.hidden_size, bert_config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(bert_config.vocab_size))
        if weight is not None:
            self.decoder.weight = weight

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x) + self.bias
        return x

class TextClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""


    def __init__(self, hidden_size = None, num_labels = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        # classifier_dropout = (
        #     config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        # )
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        # self.activation = get_activation("gelu")
        self.activation = nn.GELU()
        # self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(self.hidden_size, self.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        # x = features
        # x = self.dropout(x)
        x = self.dense(x)
        x = self.layer_norm(x)
        x = self.activation(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
        # x = self.dropout(x)
        x = self.out_proj(x)
        return x
        
        