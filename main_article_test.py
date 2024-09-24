#### block 1
import argparse
import time
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
from src.article_predict import MyModel, TextClassificationDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#### block 2

# all_df = pd.read_excel('data/article/A2_article_with_sentences_split.xlsx').fillna('')
# all_df = pd.read_excel('data/article/test_article_1108.xlsx').fillna('')
# all_df = pd.read_excel('data/article/A2_article_with_sentences_split_four_class_partial.xlsx').fillna('')
# all_df = pd.read_excel('data/article/A2_final_test_50_1113_v2.xlsx').fillna('')
# all_df = pd.read_excel('data/article/A2_article_with_sentences_split_four_class_partial.xlsx').fillna('')
# PATH = './ckpt/article/0707_300limit_6epoch_binary.pt'
# PATH = './ckpt/article/0819_400limit_3epoch_binary_short_5.pkl'
# PATH = './ckpt/article/1108_430limit_3epoch_binary_short_5.pkl'
# PATH = './ckpt/article/1112_400limit_8epoch_binary_short_type2.pkl'
# PATH = './ckpt/article/1112_500limit_8epoch_binary_short_type2.pkl'
# PATH = './ckpt/article/1113_500limit_8epoch_binary_short_type1.pkl'
# PATH = './ckpt/article/1113_500limit_3epoch_binary_short_type1_v2.pkl'

# sentence_label_list = ['無標註', '自殺與憂鬱', '無助或無望', '正向文字', '其他負向文字', '生理反應或醫療狀況', '自殺行為']
# sentence_label_list = ['無標註','自殺與憂鬱','其他負向文字', '自殺行為']
parser = argparse.ArgumentParser(description="Run model training/testing.")
parser.add_argument('fold', type=int, help='The fold number to run')
args = parser.parse_args()
fold = args.fold
test_path = f'data/final/article/A1_test_0530_fold_{fold}.xlsx'
# test_path = f'data/final/article/A1A2_test_fold_{fold}.xlsx'
all_df = pd.read_excel(test_path).fillna('')
# PATH = f'./ckpt/final/article/A1_train_augmented_fold_{fold}.pkl'
PATH = f'./ckpt/final/article/A1_train_augmented_bert_base_chinese_fold_{fold}.pkl'
# PATH = f'./ckpt/final/article/A1_train_augmented_hfl_roberta_fold_{fold}.pkl'

# PATH = f'./ckpt/final/article/A1_train_raw_type3_fold_{fold}.pkl'
# PATH = f'./ckpt/final/article/A1_train_augmented_type3_fold_{fold}.pkl'
# PATH = f'./ckpt/final/article/A1_train_raw_fold_{fold}.pkl'
# PATH = f'./ckpt/final/article/A1A2_train_raw_type3_fold_{fold}.pkl'

log_path = 'ckpt/final/article/article_A1_weighted_test.log'
# log_path = 'ckpt/final/article/article_A1A2_test.log'
# log_path = 'ckpt/final/article/article_model_test.log'

sentence_label_list = ['無標註','自殺與憂鬱', '自殺行為', '其他類型']

num_labels = 2
if "type3" in PATH:
    num_labels = 4


X = []
y = []
for index, row in all_df.iterrows():
    a_X = []
    a_y = [0.0] * num_labels
    for key in all_df:
        if key in sentence_label_list:
            a_X.append(row[key])
    crisis_level = int(row['Crisis_Level'])
    if crisis_level == 0:
        if num_labels == 2:
            y.append([1.0, 0.0])
        else:
            y.append([1.0, 0.0, 0.0, 0.0])
    elif crisis_level == 1:
        if num_labels == 2:
            y.append([1.0, 0.0])
        else:
            y.append([0.0, 1.0, 0.0, 0.0])
    elif crisis_level == 2:
        if num_labels == 2:
            if "type2" in PATH:
                y.append([0.0, 1.0])  # 32/10
            else:
                y.append([1.0, 0.0])    # 3/210
        else:
            y.append([0.0, 0.0, 1.0, 0.0])
    else:
        if num_labels == 2:
            y.append([0.0, 1.0])
        else:
            y.append([0.0, 0.0, 0.0, 1.0])
    X.append(a_X)
    # y.append(a_y)
X = np.array(X)
y = np.array(y)
print(X.shape)
print(y.shape)

# statistic about the number of sentences in each crisis level
crisis_level_count = [0, 0,0, 0]
for index, row in all_df.iterrows():
    crisis_level = int(row['Crisis_Level'])
    crisis_level_count[crisis_level] += 1
print(crisis_level_count)

#### block 3

# model_name = 'bert-base-chinese'
# model_name = 'hfl/chinese-roberta-wwm-ext'
model_name = 'hfl/chinese-bert-wwm-ext'

num_sentence_labels = len(sentence_label_list)
batch_size = 20
max_len = 512
train_epoch = 2
lr = 2e-5

#### block 4


tokenizer = BertTokenizer.from_pretrained(model_name)
config = BertConfig.from_pretrained(model_name)
config.max_length = max_len
a_model = MyModel(config, num_labels=num_labels).to(device)
a_model.load_state_dict(torch.load(PATH), strict=False)
a_model.eval()
test_dataset = TextClassificationDataset(X, y, tokenizer, 'test')
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# optimizer = AdamW(model.parameters(), lr=lr)

#### block 5


y_true = []
y_pred = []
train_total = 0.0
k = 0
threshold = 0.5
# label_1_count = 0   # count the number of label 1
for batch_data in tqdm(test_dataloader):
    tokens_tensors, segments_tensors, masks_tensors,labels  = [t.to(device) for t in batch_data]
    with torch.no_grad(): 
        loss, logits = a_model(input_ids=tokens_tensors, token_type_ids=segments_tensors, attention_mask=masks_tensors,labels=labels)
    # print(loss)
        prob = logits.data
        # if prob[1] > threshold, pred = 1, else pred = 0
        # pred = torch.zeros(prob.size(0))
        # pred[prob > threshold] = 1
        # pred[prob <= threshold] = 0
        # pred = (prob[:, 1] > threshold).float().int()
        _, pred = torch.max(prob, 1)
        # label_1_count += np.sum([1 for x in pred if x == 1])
        # _, pred = torch.max(prob, 1)
        _, truth = torch.max(labels, 1)
        # print("truth",truth)
        train_total += labels.size(0)
        y_true.append(truth)
        y_pred.append(pred)
        del tokens_tensors
        del segments_tensors
        del masks_tensors
        del labels
pred = torch.cat(y_pred)
ground_truth = torch.cat(y_true)
# print("label 1 count:", label_1_count)  
#### block 6


class_labels = [i for i in range(num_labels)]
cm = confusion_matrix(y_true=ground_truth.cpu(), y_pred=pred.cpu(), labels=class_labels)
acc = accuracy_score(y_true=ground_truth.cpu(), y_pred=pred.cpu())
f1 = f1_score(y_true=ground_truth.cpu(), y_pred=pred.cpu(), average=None, labels=class_labels)
micro_f1 = f1_score(y_true=ground_truth.cpu(), y_pred=pred.cpu(), average='micro', labels=class_labels)
macro_f1 = f1_score(y_true=ground_truth.cpu(), y_pred=pred.cpu(), average='macro', labels=class_labels)
weighted_f1 = f1_score(y_true=ground_truth.cpu(), y_pred=pred.cpu(), average='weighted', labels=class_labels)
p_score = precision_score(y_true=ground_truth.cpu(), y_pred=pred.cpu(), average=None, labels=class_labels)
micro_p = precision_score(y_true=ground_truth.cpu(), y_pred=pred.cpu(), average='micro', labels=class_labels)
macro_p = precision_score(y_true=ground_truth.cpu(), y_pred=pred.cpu(), average='macro', labels=class_labels)
weighted_p = precision_score(y_true=ground_truth.cpu(), y_pred=pred.cpu(), average='weighted', labels=class_labels)
r_score = recall_score(y_true=ground_truth.cpu(), y_pred=pred.cpu(), average=None, labels=class_labels)
micro_r = recall_score(y_true=ground_truth.cpu(), y_pred=pred.cpu(), average='micro', labels=class_labels)
macro_r = recall_score(y_true=ground_truth.cpu(), y_pred=pred.cpu(), average='macro', labels=class_labels)
weighted_r = recall_score(y_true=ground_truth.cpu(), y_pred=pred.cpu(), average='weighted', labels=class_labels)
with open(log_path, 'a') as log_file:
    print("test model name: ", PATH, file=log_file)
    print(f"fold: {fold}; time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}", file=log_file)
    print(f"accuracy: {round(acc, 3)}", file=log_file)
    # print(f"f1-score: {round(f1, 3)}", file=log_file)
    # print(f"precision_score: {round(p_score, 3)}", file=log_file)
    # print(f"recall_score: {round(r_score, 3)}", file=log_file)
    # print(f"precision_score: {[round(x, 3) for x in p_score]}", file=log_file)
    print(f"weighted_precision: {round(weighted_p, 3)}", file=log_file)
    # print(f"micro_precision: {round(micro_p, 3)}; macro_precision: {round(macro_p, 3)}", file=log_file)
    print(f"wighted_recall: {round(weighted_r, 3)}", file=log_file)
    # print(f"recall_score: {[round(x, 3) for x in r_score]}", file=log_file)
    # print(f"micro_recall: {round(micro_r, 3)}; macro_recall: {round(macro_r, 3)}", file=log_file)
    # print(f"f1-score: {[round(x, 3) for x in f1]}", file=log_file)
    print(f"weighted_f1: {round(weighted_f1, 3)}", file=log_file)
    # print(f"micro_f1: {round(micro_f1, 3)}; macro_f1: {round(macro_f1, 3)}", file=log_file)
    # print("Confusion Matrix:", file=log_file)
    # for line in cm:
    #     print(line, file=log_file)
    print("====================================", file=log_file)
    if fold == 5:
        print("===================================================================================================", file=log_file)
# add the result to the excel
# all_df['pred'] = pred.cpu().numpy()
# all_df.to_excel('data/article/A2_final_test_40_answer.xlsx', index=False)