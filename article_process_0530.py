#!/usr/bin/env python
# coding: utf-8

# In[6]:


import torch
import pandas as pd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import numpy as np
label_path = 'data/article/網路危機_A1_文章危機程度.xlsx'
# label_path = 'data/article/partial_A1_article.xlsx'
df = pd.read_excel(label_path)
df = df[['TextID','Title','Crisis_Level']]
print(df.columns)
# print(df)

for idx, val in enumerate(df['TextID']):
    if type(val) != str:
        df['TextID'][idx] = str(val)
stat = [0]*4
for idx, val in enumerate(df['Crisis_Level']):
    stat[val] += 1
print(stat)
four_label = df.values
title_list = df['Title'].values
textID_list = df['TextID'].values
sentence_path = 'data/raw/new_clean_data_all.xlsx'
s_df = pd.read_excel(sentence_path)[[ 'TextID','Title', 'Sentence', '無標註', '自殺與憂鬱',
       '無助或無望', '正向文字', '其他負向文字', '生理反應或醫療狀況', '自殺行為']]
for idx, val in enumerate(s_df['TextID']):
    if type(val) != str:
        s_df['TextID'][idx] = str(val)

sentence_label_list = ['無標註', '自殺與憂鬱', '無助或無望', '正向文字', '其他負向文字',
       '生理反應或醫療狀況', '自殺行為']
other_categories = ['無助或無望', '其他負向文字','生理反應或醫療狀況']
article_dict = {str(key): {'無標註':'', '自殺與憂鬱':'',
       '無助或無望':'', '正向文字':'', '其他負向文字':'', '生理反應或醫療狀況':'', '自殺行為':'', '其他類型':''} for key in textID_list}
problem_title = []
# count the sentence statistics for each kind of article label, which is flag
sentence_count = [[0]*7 for _ in range(4)]
for index, row in s_df.iterrows():
    article_dict[str(row['TextID'])]['label'] = df[(df['TextID']) == (row['TextID'])]['Crisis_Level']
    flag = article_dict[str(row['TextID'])]['label'].values[0]
    if flag == 3:
       flag = 1
    elif flag == 1:
       flag = 3
    article_dict[str(row['TextID'])]['Crisis_Level'] = flag
    article_dict[str(row['TextID'])]['label'] = [0.0, 0.0, 0.0, 0.0]
    article_dict[str(row['TextID'])]['label'][flag] += 1.0
    for s_label in sentence_label_list:
      if row[s_label] == 1:
        sentence_count[flag][sentence_label_list.index(s_label)] += 1
        tar = ' ' + str(row['Sentence'])
        article_dict[str(row['TextID'])][s_label] += tar
        if s_label in other_categories:
            article_dict[str(row['TextID'])]['其他類型'] += tar
        elif s_label == '正向文字':
            article_dict[str(row['TextID'])]['無標註'] += tar
        break
all_article = []
for dic in article_dict:
    single_article = pd.DataFrame([article_dict[dic]])
    single_article['TextID'] = dic
    all_article.append(single_article)
all_article_df = pd.concat(all_article,axis=0, ignore_index=True)
print(all_article_df.columns)
all_article_df['TextID'] = all_article_df['TextID'].astype(str)
all_article_df.to_excel('data/final/article/all_article_with_sentences_split_0530.xlsx', index=False)
# all_article_df.to_excel('data/article/all_article_with_sentences_split_1108.xlsx', index=False)



# In[7]:


from sklearn.model_selection import StratifiedKFold
all_df = pd.read_excel('data/final/article/all_article_with_sentences_split_0530.xlsx')
# sentence_label_list = ['無標註', '自殺與憂鬱', '無助或無望', '正向文字', '其他負向文字', '生理反應或醫療狀況', '自殺行為']
# sentence_label_list = ['無標註','自殺與憂鬱','其他負向文字', '自殺行為']
seed = 1
all_train_df = []
all_test_df = []
# 5-fold cross validation
skf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
for train_index, test_index in skf.split(all_df, all_df['Crisis_Level']):
    train_df = all_df.iloc[train_index]
    test_df = all_df.iloc[test_index]
    all_train_df.append(train_df)
    all_test_df.append(test_df)
    
for i in range(5):
    # save train and test data
    all_train_df[i].to_excel(f'data/final/article/A1_train_0530_fold_{i+1}.xlsx', index=False)
    all_test_df[i].to_excel(f'data/final/article/A1_test_0530_fold_{i+1}.xlsx', index=False)


# In[8]:
for i in range(1,6):
    without_test_df = pd.read_excel(f'data/final/article/A1_train_0530_fold_{i}.xlsx')
    df_crisis_level_0 = without_test_df.loc[without_test_df['Crisis_Level'] == 0]
    df_augmentation_0 = df_crisis_level_0.sample(n=100, random_state=seed)
    df_remaining = without_test_df.drop(df_augmentation_0.index) 

    # 當作Augmentation的資料，將Crisis_Level為0(無危機)的資料取出
    df_augmentation_0.to_excel(f'data/final/article/A1_be_augment_0_0530_fold_{i}.xlsx', index=False)

    df_crisis_level_1 = df_remaining.loc[without_test_df['Crisis_Level'] == 1]
    df_augmentation_1 = df_crisis_level_1.sample(n=100, random_state=1)
    df_remaining_v2 = df_remaining.drop(df_augmentation_1.index) 
    df_augmentation_1.to_excel(f'data/final/article/A1_be_augment_1_0530_fold_{i}.xlsx', index=False)


    df_remaining_v2.to_excel(f'data/final/article/A1_train_moveAug_0530_fold_{i}.xlsx', index=False)


# In[13]:

# # In[1]:


# import pandas as pd
# A1_df = pd.read_excel('data/article/train_article_1108.xlsx')
# A2_df = pd.read_excel('data/article/A2_final_wo_test_50_1113_v2.xlsx')
# print(A1_df.columns)
# print(A2_df.columns)
# # retrieve the A2 columns of A1
# A1_df = A1_df[A2_df.columns]
# # concat A1 and A2 and save
# A1_A2_df = pd.concat([A1_df, A2_df], axis=0, ignore_index=True)
# # statistic of A1_A2_df Crisis_Level
# statsssss = [0]*4
# for index, row in A1_A2_df.iterrows():
#     statsssss[row['Crisis_Level']] += 1
# print(statsssss)

# A1_A2_df.to_excel('data/article/train_A1_A2_1113_v2.xlsx', index=False)


