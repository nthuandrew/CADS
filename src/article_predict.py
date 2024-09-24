import re
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel,  BertForSequenceClassification
from transformers import BertTokenizer, BertConfig
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import itertools

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sentence_label_list = ['無標註','自殺與憂鬱', '自殺行為', '其他類型']
num_sentence_labels = len(sentence_label_list)
batch_size = 4
max_len = 512
train_epoch = 4
lr = 2e-5

import re

def split_sentence(sentence, maxlen=30):
    # 先以換行符進行分割
    sent_parts = re.split('(\n)', sentence)
    new_parts = []
    
    for part in sent_parts:
        # 檢查每個分割後的部分是否超過maxlen
        while len(part) > maxlen:
            # 先嘗試以句號和空格進行分割
            sub_parts = re.split('(。| )', part, 1)  # 只分割第一個匹配的分隔符
            if len(sub_parts) > 1 and len(sub_parts[0]) <= maxlen:
                # 如果第一部分的長度合適，則將其添加到new_parts
                new_parts.append(sub_parts[0] + (sub_parts[1] if len(sub_parts) > 1 else ''))
                # 處理剩餘的部分
                part = ''.join(sub_parts[2:])
            else:
                # 如果沒有句號或空格可以分割，則嘗試逗號
                sub_sub_parts = re.split('(，|,)', part, 1)  # 只分割第一個匹配的分隔符
                if len(sub_sub_parts) > 1 and len(sub_sub_parts[0]) <= maxlen:
                    new_parts.append(sub_sub_parts[0] + (sub_sub_parts[1] if len(sub_sub_parts) > 1 else ''))
                    part = ''.join(sub_sub_parts[2:])
                else:
                    # 如果沒有逗號可以分割，則直接分割到maxlen
                    new_parts.append(part[:maxlen])
                    part = part[maxlen:]
                    
        # 將最後的部分（或者原本就沒有超過maxlen的部分）添加到new_parts
        if part:
            new_parts.append(part)
    
    return new_parts
    
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, mode, max_len=max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_len = max_len


    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx].tolist()
        # print(type(text))
        # print(text)
        label = self.labels[idx]
        # if self.mode == 'test':
        #     label_tensor = None
        # else:
        label_tensor = torch.tensor(label)
        tokenized = self.tokenizer(text, padding='max_length', truncation="longest_first", return_tensors='pt', max_length=self.max_len)
        # tokenized = self.tokenizer(text, padding=512, truncation=True, return_tensors='pt')
        tokens_tensor = tokenized['input_ids']
        segments_tensor = tokenized['token_type_ids']
        mask_tensor = tokenized['attention_mask']
        
        # inputs = [inp for inp in inputs]
        # print(inputs)
        return (tokens_tensor, segments_tensor, mask_tensor, label_tensor)

class MyModel(BertPreTrainedModel):
    def __init__(self, config, num_labels ):
        super(MyModel, self).__init__(config)
        self.num_labels = num_labels
        self.embedding = []
        self.bert = BertModel(config)
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=2))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 4, kernel_size=2, stride=1, padding=1),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=2))
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(int(np.ceil(num_sentence_labels/4))*4*192, num_labels))
        self.nn1 = nn.Linear(num_sentence_labels*self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.nn2 = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.pooling_strategy = 'reduce_mean'
        self.class_weights = [1]*num_labels
        self.weights = torch.tensor(self.class_weights, dtype=torch.float)
    def forward(self, input_ids, token_type_ids=None, attention_mask=None,labels=None):
        for i in range(num_sentence_labels):
            a_kind_of_input_ids = input_ids[:, i, :]
            a_kind_of_token_type_ids = token_type_ids[:,i,:]
            a_kind_of_attention_mask = attention_mask[:,i,:]
            a_kind_of_embedding = self.bert(input_ids=a_kind_of_input_ids, token_type_ids=a_kind_of_token_type_ids,attention_mask=a_kind_of_attention_mask)
            encoder_hidden_states = a_kind_of_embedding.pooler_output.to(device)
            # print("encoder_hidden_states.shape:",encoder_hidden_states.shape)
            # if self.pooling_strategy == 'cls':
            #     cls_vector = encoder_hidden_states[:, 0, :]
            # elif self.pooling_strategy == 'reduce_mean':
            #     cls_vector = encoder_hidden_states.sum(axis=1) / attention_mask.sum(axis=-1).unsqueeze(-1)
            # print(cls_vector.shape)
            self.embedding.append(encoder_hidden_states)
        # concat_embedding = torch.cat(self.embedding, dim=1)
        concat_embedding = torch.stack(self.embedding).permute(1, 0, 2)
        to_shape = (concat_embedding.shape)
        concat_embedding = concat_embedding.reshape(to_shape[0], 1, to_shape[1], to_shape[2])
        self.embedding = []
        # print(concat_embedding.shape)
        # outputs = self.bert(input_ids, token_type_ids=token_type_ids,attention_mask=attention_mask)
        # outputs_concat = concat_embedding.reshape(-1,)
        
        # logits = self.nn1(concat_embedding)
        # logits = self.relu(logits)
        # logits = self.nn2(logits)
        logits = self.conv1(concat_embedding)
        logits = self.conv2(logits)
        to_shape_logits = logits.shape
        logits = logits.view(-1, to_shape_logits[1]*to_shape_logits[2]*to_shape_logits[3])
        logits = self.dense(logits)
        logits = self.softmax(logits)
        # print("logits.shape:",logits.shape)
        loss = None
        # print(logits.shape)
        # print(labels.shape)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.weights.to(device))
            loss_fct = nn.CrossEntropyLoss()
            loss_ = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            loss = loss_
        # print(logits, loss)
        output = (logits,) # + outputs[2:]
        if loss is not None:
            return ((loss,) + output)
        else:
            return output
    
    def get_predictions(self, test_dataloader):
        y_true = []
        y_pred = []
        train_total = 0.0
        for batch_data in (test_dataloader):
            tokens_tensors, segments_tensors, masks_tensors,labels  = [t.to(device) for t in batch_data]
            with torch.no_grad(): 
                loss, logits = self.forward(input_ids=tokens_tensors, token_type_ids=segments_tensors, attention_mask=masks_tensors,labels=labels)
            # print(loss)
                prob = logits.data
                _, pred = torch.max(prob, 1)
                _, truth = torch.max(labels, 1)
                train_total += labels.size(0)
                y_true.append(truth)
                y_pred.append(pred)
                del tokens_tensors
                del segments_tensors
                del masks_tensors
                del labels
        pred = torch.cat(y_pred)
        ground_truth = torch.cat(y_true)
        return pred
        # Crisis_level = article_label_list[pred[0]]





