import torch
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from data.GV import *

num_encoder_hidden = 768

class BertForClassification(BertPreTrainedModel):
    def __init__(self, config, *model_args, **model_kwargs):
        super(BertForClassification, self).__init__(config, *model_args, **model_kwargs)
        self.num_labels = config.num_labels
        self.MAX_LENGTH = config.max_length
        self.DEVICE = model_kwargs['device']
        self.pooling_strategy = model_kwargs['pooling_strategy']
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.relu = nn.ReLU()
        self.output_base = nn.Linear(num_encoder_hidden, self.num_labels)
        self.init_weights()
    def forward(self, input_ids, token_type_ids=None, attention_mask=None,labels=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids,attention_mask=attention_mask)
        sequence_output = outputs[0]
        encoder_hidden_states = sequence_output.to(self.DEVICE)

        if self.pooling_strategy == 'cls':
            cls_vector = encoder_hidden_states[:, 0, :]
        elif self.pooling_strategy == 'reduce_mean':
            cls_vector = encoder_hidden_states.sum(axis=1) / attention_mask.sum(axis=-1).unsqueeze(-1)
        else:
            print('>>>>> ErrorL Wrong pooling strategy! We now only support "reduce_mean" and "cls". >>>>>')
            return
        logits = self.output_base(cls_vector)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss_ = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss = loss_

        output = (logits,) + outputs[2:]
        if loss is not None:
            return ((loss,) + output)
        else:
            return output