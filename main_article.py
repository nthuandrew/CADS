#### block 1
import argparse
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel,  BertForSequenceClassification
from transformers import BertTokenizer, BertConfig
from transformers import (
  BertTokenizerFast,
  AutoModel,
)
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import itertools
from module.util import seed_torch
from src.article_predict import MyModel, TextClassificationDataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# PATH = './ckpt/article/0819_430limit_3epoch_binary_short_5.pkl'
# PATH = './ckpt/article/1108_400limit_4epoch_binary_short_type1.pkl'
# model_name = 'bert-base-chinese'
# model_name = 'hfl/chinese-roberta-wwm-ext'
model_name = 'hfl/chinese-bert-wwm-ext'

# model_name = 'ckiplab/bert-base-chinese-ner'
sentence_label_list = ['無標註','自殺與憂鬱', '自殺行為', '其他類型']
# sentence_label_list = ['無標註','自殺與憂鬱', '自殺行為', '其他類型']
num_sentence_labels = len(sentence_label_list)
batch_size = 4
max_len = 512
train_epoch = 8
lr = 2e-5
parser = argparse.ArgumentParser(description="Run model training/testing.")
parser.add_argument('fold', type=int, help='The fold number to run')
args = parser.parse_args()
fold = args.fold
seed = 1
seed_torch(seed=seed)
train_path = f'data/final/article/A1_train_augmented_0530_fold_{fold}.xlsx'   # A1 only, augmentation
# train_path = f'data/final/article/A1_train_0530_fold_{fold}.xlsx'    # A1 only, no augmentation
# train_path  = f'data/final/article/A1A2_train_augmented_fold_{fold}.xlsx'   # A1A2, augmentation
# train_path = f'data/final/article/A1A2_train_fold_{fold}.xlsx'   # A1A2, no augmentation
# PATH = f'./ckpt/final/article/A1_train_augmented_hfl_roberta_fold_{fold}.pkl'
PATH = f'./ckpt/final/article/A1_train_augmented_bert_base_chinese_fold_{fold}.pkl'
# PATH = f'./ckpt/final/article/A1A2_train_augmented_type3_fold_{fold}.pkl'
# PATH = f'./ckpt/final/article/A1_train_augmented_type3_fold_{fold}.pkl'

num_labels = 2

if "type3" in PATH:
    num_labels = 4
ratio = 0.99

#### block 2

# all_df = pd.read_excel('data/article/train_augmented_1_v4_short_v3.xlsx').fillna('')
# all_df = pd.read_excel('data/article/train_augmented_short_1108.xlsx').fillna('')
# all_df = pd.read_excel('data/article/train_augmented_short_1113_SSL.xlsx').fillna('')
# all_df = pd.read_excel('data/article/train_augmented_1113_SSL_v2.xlsx').fillna('')
all_df = pd.read_excel(train_path).fillna('')
# sentence_label_list = ['無標註', '自殺與憂鬱', '無助或無望', '正向文字', '其他負向文字', '生理反應或醫療狀況', '自殺行為']
# sentence_label_list = ['無標註','自殺與憂鬱','其他負向文字', '自殺行為']

X = []
y = []
for index, row in all_df.iterrows():
    a_X = []
    a_y = [0.0, 0.0]
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
X = np.array(X)
y = np.array(y)
print(X.shape)
print(y.shape)

#### block 3
# limit_amount = 300 # augmentation, A1 only, type1(3/210)
# limit_amount = 100 # no augmentation, A1 only, type3(four class)
limit_amount = 300 # augmentation, A1 only, type3(four class)
# limit_amount = 250 # no augmentation, A1 only, type2 (32/10)
# limit_amount = 600 # augmentation, A1 only, type2 (32/10)
# limit_amount = 500 # augmentation, A1A2, type1 (3/210) and type3(four class)
# limit_amount = 1000 # augmentation, A1A2, type2 (32/10)
# limit_amount = 150 # no augmentation, A1A2, type1 (3/210) and type3(four class)
# limit_amount = 400 # no augmentation, A1A2, type2 (32/10)

limit = [limit_amount]*num_labels
count = [0]*num_labels
partial_X = []
partial_y = []
for idx, y_ in enumerate(y):
    crisis_level = np.where(y_ == 1)[0][0]
    if count[crisis_level] < limit[crisis_level]:
        count[crisis_level] += 1
        partial_X.append(X[idx])
        partial_y.append(y_)
partial_X = np.array(partial_X)
partial_y = np.array(partial_y)
print(partial_X.shape)
print(partial_y.shape)
# print statistics of partial_y
print("Distribution  0    1    2    3")
print(np.sum(partial_y, axis = 0))


#### block 4

X_train, X_test, y_train, y_test = train_test_split(partial_X, partial_y, random_state=seed, train_size=ratio)
X_val_test = X_test
y_val_test = y_test
X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, random_state=seed, train_size=0.5)
print(len(X_train), len(X_valid), len(X_test))
print("Distribution  0    1    2    3")
print("train: ",np.sum(y_train, axis = 0))
print("valid: ",np.sum(y_valid, axis = 0))
print("test:  ", np.sum(y_test, axis = 0))

#### block 5


#### block 6
# tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
# model = AutoModel.from_pretrained('ckiplab/bert-base-chinese-ner')

tokenizer = BertTokenizer.from_pretrained(model_name)
config = BertConfig.from_pretrained(model_name)
config.max_length = max_len
model = MyModel(config, num_labels=num_labels).to(device)

train_dataset = TextClassificationDataset(X_train, y_train, tokenizer, 'train')
val_dataset = TextClassificationDataset(X_valid, y_valid, tokenizer, 'valid')
test_dataset = TextClassificationDataset(X_test, y_test, tokenizer, 'test')
val_test_dataset = TextClassificationDataset(X_val_test, y_val_test, tokenizer, 'test')

train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, batch_size=2)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
val_test_dataloader = DataLoader(val_test_dataset, batch_size=batch_size)

optimizer = AdamW(model.parameters(), lr=lr)

#### block 7

# Training
model.train()
for epoch in range(train_epoch):  # number of epochs
    print("epoch ", epoch+1)
    running_train_loss = 0.0
    running_train_correct = 0.0
    train_total = 0.0
    for batch_data in tqdm(train_dataloader):
        tokens_tensors, segments_tensors, masks_tensors, labels = [t.to(device) for t in batch_data]
        optimizer.zero_grad()
        loss, logits = model(input_ids=tokens_tensors, token_type_ids=segments_tensors, attention_mask=masks_tensors,labels=labels)
        # print(loss)
        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        running_train_loss += loss.item()
        prob = logits.data
        _, pred = torch.max(prob, 1)
        _, truth = torch.max(labels, 1)
        train_total += labels.size(0)
        running_train_correct += (pred == truth).sum().item()
        del tokens_tensors
        del segments_tensors
        del masks_tensors
        del labels
    train_acc = running_train_correct / train_total
    avg_running_train_loss = running_train_loss / len(train_dataloader)
    print('Train>>>[epoch %d] loss: %.3f, acc: %.3f' %
        (epoch + 1, avg_running_train_loss, train_acc))
    print("Validation: ")
    running_val_loss = 0.0
    running_val_correct = 0.0
    val_total = 0.0
    y_true = []
    y_pred = []
    for batch_data in tqdm(val_dataloader):
        tokens_tensors, segments_tensors, masks_tensors, labels = [t.to(device) for t in batch_data]
        with torch.no_grad(): 
            loss, logits = model(input_ids=tokens_tensors, token_type_ids=segments_tensors, attention_mask=masks_tensors,labels=labels)
            running_val_loss += loss.item()
            prob = logits.data
            _, pred = torch.max(prob, 1)
            _, truth = torch.max(labels, 1)
            val_total += labels.size(0)
            running_val_correct += (pred == truth).sum().item()
            y_true.append(truth)
            y_pred.append(pred)
            del tokens_tensors
            del segments_tensors
            del masks_tensors
            del labels
    val_acc = running_val_correct / val_total
    avg_running_val_loss = running_val_loss / len(val_dataloader)
    print('Validation>>>[epoch %d] loss: %.3f, acc: %.3f' %
        (epoch + 1, avg_running_val_loss, val_acc))
    # confusion matrix
    class_labels = [i for i in range(num_labels)]
    cm = confusion_matrix(y_true=torch.cat(y_true).cpu(), y_pred=torch.cat(y_pred).cpu(), labels=class_labels)
    print(cm)
        
#### block 8


y_true = []
y_pred = []
train_total = 0.0
for batch_data in tqdm(test_dataloader):
    tokens_tensors, segments_tensors, masks_tensors,labels  = [t.to(device) for t in batch_data]
    with torch.no_grad(): 
        loss, logits = model(input_ids=tokens_tensors, token_type_ids=segments_tensors, attention_mask=masks_tensors,labels=labels)
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

#### block 9
class_labels = [i for i in range(num_labels)]
cm = confusion_matrix(y_true=ground_truth.cpu(), y_pred=pred.cpu(), labels=class_labels)
acc = accuracy_score(y_true=ground_truth.cpu(), y_pred=pred.cpu())
f1 = f1_score(y_true=ground_truth.cpu(), y_pred=pred.cpu(), average='weighted')
print("accuracy: ", round(acc, 3))
print("f1-score: ", round(f1, 3))
print(cm)
#### block 10

torch.save(model.state_dict(), PATH)



