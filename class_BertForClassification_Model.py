import torch
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from data.GV import *

num_encoder_hidden = 768
num_attention_hidden = 350
num_hops = 1
num_classifier_hidden = 2000

class BertForClassification(BertPreTrainedModel):
    def __init__(self, config, *model_args, **model_kwargs):
        super(BertForClassification, self).__init__(config, *model_args, **model_kwargs)
        self.num_labels = config.num_labels
        self.MAX_LENGTH = config.max_length
        self.DEVICE = model_kwargs['device']
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.relu = nn.ReLU()
        # self.output = nn.Sequential(
        #     nn.Linear(num_hops * num_encoder_hidden, num_classifier_hidden),
        #     nn.ReLU(),
        #     nn.Linear(num_classifier_hidden, self.num_labels)
        # )
        # Debug
        self.output = nn.Sequential(
            # nn.Linear(num_hops * num_encoder_hidden, num_classifier_hidden),
            # nn.ReLU(),
            # nn.LayerNorm(num_hops * num_encoder_hidden, eps=1e-12),
            # nn.Linear(num_hops * num_encoder_hidden, self.num_labels)
            # nn.LayerNorm(num_hops * num_encoder_hidden, eps=1e-12),
            nn.Linear(self.MAX_LENGTH*num_hops, self.num_labels)
        )

        self.output_base = nn.Linear(num_encoder_hidden, self.num_labels)
        self.output_base2 = nn.Linear(num_encoder_hidden * self.MAX_LENGTH, self.num_labels)
        # self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()
    def forward(self, input_ids, token_type_ids=None, attention_mask=None,labels=None, lengths= None, output_attention=False):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids,attention_mask=attention_mask)
        sequence_output = outputs[0]
        encoder_hidden_states = sequence_output.to(self.DEVICE)
        ############## Baseline1 #####################
        cls_vector = encoder_hidden_states[:, 0, :]
        logits = self.output_base(cls_vector)
        ############## END #####################

        ############## Baseline2 #####################
        # cls_vector = encoder_hidden_states.view(-1, num_encoder_hidden * MAX_LENGTH)
        # logits = self.output_base2(cls_vector)
        ############## END #####################

        ############## DW #####################
        # # outputs = self.dropout(encoder_hidden_states)
        # lengths = lengths.to(device)
        # encoder_hidden_states = encoder_hidden_states.transpose(1,2)
        # attention_weights = self.attention(encoder_hidden_states.to(device), lengths.to(device))
        # attention_weights = self.relu(attention_weights)
        # # Debug
        # # attention_weights = self.relu(attention_weights)
        # sentence_embedding_ = torch.bmm(encoder_hidden_states.transpose(1,2), attention_weights)
        # # sentence_embedding_ = (batch_size, sentence len, num_hops)
        # # Debug
        # # sentence_embedding_ = self.relu(sentence_embedding_)
        # sentence_embedding_  = sentence_embedding_ .pow(2)
        # sentence_embedding_norm = sentence_embedding_.sum(dim=2).unsqueeze(2)
        # # print(sentence_embedding_.shape, sentence_embedding_norm.shape)
        # sentence_embedding_ = torch.div(sentence_embedding_, sentence_embedding_norm)
        # output_sentence_embedding = sentence_embedding_
        # # print(output_sentence_embedding)
        # # output_sentence_embedding = nn.functional.softmax(sentence_embedding_ , dim=2).transpose(1,2)
        # sentence_embedding = sentence_embedding_.view(-1, self.attention.num_hops * MAX_LENGTH)
        # logits = self.output(sentence_embedding)
        ############## END #####################

        loss = None
        if labels is not None:
            B = logits.view(-1, self.num_labels).size(0)
            loss_fct = CrossEntropyLoss()
            loss_ = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            # AAT = torch.bmm(attention_weights.transpose(1,2), attention_weights).to(device)
            # I = torch.eye(num_hops).unsqueeze(0).repeat(B, 1, 1).to(device)
            # penalization_term = (torch.norm(AAT - I) / B).to(device)
            # loss = loss_ + penalization_term
            loss = loss_

        output = (logits,) + outputs[2:]
        if loss is not None:
            return ((loss,) + output)
        else:
            if output_attention:
                return output, output_sentence_embedding
            return output