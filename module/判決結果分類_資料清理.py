## 判決結果分類_資料清理
# In[1]:
# 讀取標註檔案
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import json


from importlib import import_module, reload
fp = reload(import_module('custody-prediction-dataset.custody-prediction-data-preparation.lib.filepath_process'))
get_path_from_json = fp.get_path_from_json
path = get_path_from_json('labels_full_murphy', 'xlsx')
df = pd.read_excel(path)
# df = pd.read_excel('/Users/juanmurphy/Documents/ENV/custody-prediction-modeling/dataset/custody-prediction-dataset/labels_full(2616)_murphy_v3.xlsx')
# df = pd.read_excel('/Users/juanmurphy/Documents/ENV/custody-prediction-modeling/dataset/custody-prediction-dataset/labels_full_rf.xlsx')

# 讀取中性句子
from importlib import import_module, reload
fp = reload(import_module('custody-prediction-dataset.custody-prediction-data-preparation.lib.filepath_process'))
get_path_from_json = fp.get_path_from_json
path = get_path_from_json('neutral_snetence_from_judgements', 'xlsx')
df_neu = pd.read_excel(path)
# df_neu = pd.read_excel('/Users/juanmurphy/Documents/ENV/custody-prediction-modeling/dataset/custody-prediction-dataset/custody-prediction-data-preparation/neutral_snetence_from_judgements.xlsx')

# for 判決結果分類
# 取出 Result == 'a', 'b', 'c' 的案件
keep_result = ['a', 'b', 'c']
df = df.loc[df['Result'].isin(keep_result)]

# 取出 Willingness == 'a' 的案件
# select only Willingness == 'a'
df = df.loc[ df['Willingness'] == 'a']

# Debug
# 清理後之統計數量
# print("total conut: ", len(df))
# categorical = ['Result','Willingness','Type', 'AK','RK','AN','RN']
# for i_column in categorical:
#     print('%s count: %s' % (i_column, Counter(df[i_column])))
# 酌定的數量
# df_tmp2 = df[df['Type']=='a']
# print("total conut: ", len(df_tmp2))
# categorical = ['Result','Willingness','Type', 'AK','RK','AN','RN']

# for i_column in categorical:
#     print('%s count: %s' % (i_column, Counter(df_tmp2[i_column])))

# 改定的數量
# df_tmp2 = df[df['Type']=='b']
# print("total conut: ", len(df_tmp2))
# categorical = ['Result','Willingness','Type', 'AK','RK','AN','RN']
# for i_column in categorical:
#     print('%s count: %s' % (i_column, Counter(df_tmp2[i_column])))

df_neu = df_neu[df_neu['ID'].isin(df['ID'])]
# 儲存清理過的 dataset (reasoning, 供判決結果分類使用)
from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df, test_size=0.2, random_state=None)
df_train_neu, df_test_neu = train_test_split(df_neu, test_size=0.2, random_state=None)
from importlib import import_module, reload
fp = reload(import_module('custody-prediction-dataset.custody-prediction-data-preparation.lib.filepath_process'))
save_path_to_json = fp.save_path_to_json
# save training set
total_number = len(df_train)
path = save_path_to_json('for_classifier_training/judgment_result_train', 'msg', str(total_number))
df_train.to_msgpack(path)
path = save_path_to_json('for_classifier_training/judgment_result_train_neu', 'msg', str(total_number))
df_train_neu.to_msgpack(path)
# save testing set
total_number = len(df_test)
path = save_path_to_json('for_classifier_training/judgment_result_test', 'msg', str(total_number))
df_test.to_msgpack(path)
path = save_path_to_json('for_classifier_training/judgment_result_test_neu', 'msg', str(total_number))
df_test_neu.to_msgpack(path)
# 類別標記轉成 one-hot encode
categorical = ['Result','Willingness','AK','RK','AN','RN', 'Type']
df2 = df
for i_column in categorical:
    # split labels, expand and stack it as new index levels
    df_tmp2 = df[i_column].str.split('|', expand=True).stack()
    # one-hot encode, groupby level 0 and sum it
    df_tmp2 = pd.get_dummies(df_tmp2).groupby(level=0).sum()
    # search for multiple labels (mutiple ones)
    df_tmp2.apply(lambda x: print(x) if sum(x) != 1 else x, axis=1)
    # apply to np.array
    df2[i_column] = df_tmp2.apply(lambda x: tuple(x), axis=1).apply(np.array)
# 儲存onehot 後的檔案
# 如果不在分開test跟train之前先做onehot，就有可能遇到當兩邊轉成onehot 後長度不一致的問題。由於sampling的關係，可能具有某個選項的data 只出現在training set，而沒出現在test set，此時兩邊做onehot就會不一致，test set 就會少一個選項。而且也無法後來在補，因為可能是中間某個選項不見了，導致onehot的順序都不一樣了。

from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df2, test_size=0.2, random_state=None)
from importlib import import_module, reload
fp = reload(import_module('custody-prediction-dataset.custody-prediction-data-preparation.lib.filepath_process'))
save_path_to_json = fp.save_path_to_json
# save training set
total_number = len(df_train)
path = save_path_to_json('for_classifier_training/judgment_result_onehot_train_murphy', 'msg', str(total_number))
df_train.to_msgpack(path)
# save testing set
total_number = len(df_test)
path = save_path_to_json('for_classifier_training/judgment_result_onehot_test_murphy', 'msg', str(total_number))
df_test.to_msgpack(path)

# for 有利不利句子分類
import numpy as np

def output_to_list(content, content_list):
    #print(type(pd.Series()))
    if type(content) is type(pd.Series()):
        #print(type(pd.Series()))
        content.apply(output_to_list, content_list=content_list)
    elif type(content) is float:
        if not np.isnan(content):
            content_list.append(content)
    elif content is not np.nan:
        content_list.append(content)

applicant_advantage_column = df.columns[df.columns.to_series().str.contains('AA')].tolist()
respondent_advantage_column = df.columns[df.columns.to_series().str.contains('RA')].tolist()
applicant_disadvantage_column = df.columns[df.columns.to_series().str.contains('AD')].tolist()
respondent_disadvantage_column = df.columns[df.columns.to_series().str.contains('RD')].tolist()
neutral_column = df_neu.columns[df_neu.columns.to_series().str.contains('neutral')].tolist()

advantage_column = applicant_advantage_column + respondent_advantage_column
disadvantage_column = applicant_disadvantage_column + respondent_disadvantage_column
# training sentence set
advantage_train_list=[]
disadvantage_train_list=[]
neutral_train_list=[]

df_train.loc[:,advantage_column].apply(output_to_list, content_list=advantage_train_list)
df_train.loc[:,disadvantage_column].apply(output_to_list, content_list=disadvantage_train_list)
df_train_neu.loc[:,neutral_column].apply(output_to_list, content_list=neutral_train_list)

# testing sentence set
advantage_test_list=[]
disadvantage_test_list=[]
neutral_test_list=[]

df_test.loc[:,advantage_column].apply(output_to_list, content_list=advantage_test_list)
df_test.loc[:,disadvantage_column].apply(output_to_list, content_list=disadvantage_test_list)
df_test_neu.loc[:,neutral_column].apply(output_to_list, content_list=neutral_test_list)

from sklearn.utils import shuffle
# training 
advantage_train_list_shuffled = shuffle(advantage_train_list, random_state=1234)
disadvantage_train_list_shuffled = shuffle(disadvantage_train_list, random_state=1234)
# reduced training neutral sentence
n_neutral_samples = int((len(advantage_train_list_shuffled)+len(disadvantage_train_list_shuffled))/2)
neutral_train_list_shuffled = shuffle(neutral_train_list, random_state=1234)[:n_neutral_samples]
# testing 
advantage_test_list_shuffled = shuffle(advantage_test_list, random_state=1234)
disadvantage_test_list_shuffled = shuffle(disadvantage_test_list, random_state=1234)
# reduced testing neutral sentence
n_neutral_samples = int((len(advantage_test_list_shuffled)+len(disadvantage_test_list_shuffled))/2)
neutral_test_list_shuffled = shuffle(neutral_test_list, random_state=1234)[:n_neutral_samples]

# 輸出有利不利文字到檔案

def writetofile(output_path, content_list):
    import os

    list_path=os.path.expanduser(output_path)

    with open(list_path, 'w', encoding='utf-8') as f:
        for index, data in enumerate(content_list):
            f.write(str(index) + ' ' + data + '\n')
            f.write('\n')

from importlib import import_module, reload
fp = reload(import_module('custody-prediction-dataset.custody-prediction-data-preparation.lib.filepath_process'))
save_path_to_json = fp.save_path_to_json

# save training sentence
total_number = len(advantage_train_list_shuffled)
path = save_path_to_json('for_feature_extraction/sentence_advantages_train_murphy', 'msg', str(total_number))
pd.DataFrame(advantage_train_list_shuffled).to_msgpack(path)

total_number = len(disadvantage_train_list_shuffled)
path = save_path_to_json('for_feature_extraction/sentence_disadvantages_train_murphy', 'msg', str(total_number))
pd.DataFrame(disadvantage_train_list_shuffled).to_msgpack(path)

total_number = len(neutral_train_list_shuffled)
path = save_path_to_json('for_feature_extraction/sentence_neutrals_train_murphy', 'msg', str(total_number))
pd.DataFrame(neutral_train_list_shuffled).to_msgpack(path)

# save testing sentence
total_number = len(advantage_test_list_shuffled)
path = save_path_to_json('for_feature_extraction/sentence_advantages_test_murphy', 'msg', str(total_number))
pd.DataFrame(advantage_test_list_shuffled).to_msgpack(path)

total_number = len(disadvantage_test_list_shuffled)
path = save_path_to_json('for_feature_extraction/sentence_disadvantages_test_murphy', 'msg', str(total_number))
pd.DataFrame(disadvantage_test_list_shuffled).to_msgpack(path)

total_number = len(neutral_test_list_shuffled)
path = save_path_to_json('for_feature_extraction/sentence_neutrals_test_murphy', 'msg', str(total_number))
pd.DataFrame(neutral_test_list_shuffled).to_msgpack(path)
