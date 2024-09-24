#!/usr/bin/env python
# coding: utf-8

# In[1]:

from tqdm import tqdm
import torch
import pandas as pd
import numpy as np

for i in tqdm(range(1, 6)):
    augment_df1 = pd.read_excel(f'data/final/article/A1A2_be_augment_1_0530_fold_{i}.xlsx').fillna("")
    augment_df2 = pd.read_excel(f'data/final/article/A1A2_be_augment_0_0530_fold_{i}.xlsx').fillna("")
    origin_df = pd.read_excel(f'data/final/article/A1A2_train_moveAug_0530_fold_{i}.xlsx').fillna("")
    # origin_df = pd.read_excel('data/article/train_A1_A2_1113_v2.xlsx').fillna("")
    augment_df = pd.concat([augment_df1, augment_df2],axis=0, ignore_index=True)

    # statistic of origin_df

    print(origin_df['Crisis_Level'].value_counts())


    # In[2]:


    print(augment_df['自殺與憂鬱'][2].split())
    augment = augment_df['自殺與憂鬱'].dropna()
    total = 0
    all_negative_sentence = []
    for idx,para in enumerate(augment):
        # print(para.split('_')[1:])
        # if idx != 1:
        total += len(para.split(' '))
        all_negative_sentence += para.split(' ')

    # remove all empty string in all_negative_sentence
    all_negative_sentence = list(filter(None, all_negative_sentence))
    sentence_len = len(all_negative_sentence)
    print("total length of all negative sentence: ", sentence_len)

    # In[3]:


    level_3_df = origin_df[origin_df['Crisis_Level'] == 3]
    level_2_df = origin_df[origin_df['Crisis_Level'] == 2]
    level_3_len = len(level_3_df)
    level_2_len = len(level_2_df)


    # In[4]:


    # duplicate level_3_df from len = level_3_len to len = 100
    augment_amount_of_level_2 = 250
    level_3_df = pd.concat([level_3_df]*int(sentence_len/level_3_len), ignore_index=True)
    # select the first 84+178 sentence
    level_3_df = level_3_df.iloc[:sentence_len-augment_amount_of_level_2]

    level_2_df = pd.concat([level_2_df]*int(sentence_len/level_3_len), ignore_index=True)
    # select the first 84+178 sentence
    level_2_df = level_2_df.iloc[:augment_amount_of_level_2]



    # randomly choose a sentence from all_negative_sentence and replace with a sentence in level_3_df
    import random
    seed = 1
    random.seed(seed)

    for idx, row in level_3_df.iterrows():
        para = row['自殺與憂鬱'].split(' ')
        random_idx_1 = np.random.randint(0, len(para))  
        random_idx_2 = np.random.randint(0, len(all_negative_sentence))    
        para[random_idx_1] = all_negative_sentence[random_idx_2]
        all_negative_sentence.pop(random_idx_2)
        row['自殺與憂鬱'] = ' '.join(para)
        level_3_df.iloc[idx] = row



    for idx, row in level_2_df.iterrows():
        para = row['自殺與憂鬱'].split(' ')
        random_idx_1 = np.random.randint(0, len(para))  
        random_idx_2 = np.random.randint(0, len(all_negative_sentence))    
        para[random_idx_1] = all_negative_sentence[random_idx_2]
        all_negative_sentence.pop(random_idx_2)
        row['自殺與憂鬱'] = ' '.join(para)
        level_2_df.iloc[idx] = row

    augmented_df = pd.concat([origin_df, level_3_df, level_2_df],axis=0, ignore_index=True)


    # In[9]:

    # print(augmented_df['Crisis_Level'].value_counts())

    # balance augmented_df by crisis level, constrain the amount of each level under 600 
    # maxima = 500
    # augmented_df_v2 = augmented_df.groupby('Crisis_Level').head(maxima).reset_index(drop=True)
    # print(augmented_df_v2['Crisis_Level'].value_counts())
    # shuffle the augmented_df without index and set the random_state
    augmented_df_v2 = augmented_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    print(augmented_df_v2['Crisis_Level'].value_counts())
    # augmented_df.to_excel('data/article/train_augmented_1_v4.xlsx')
    augmented_df_v2.to_excel(f'data/final/article/A1A2_train_augmented_fold_{i}.xlsx')



