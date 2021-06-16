# 判決結果分類_斷詞

# 讀取標注資料
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.utils import shuffle
from importlib import import_module, reload

fp = reload(import_module('custody-prediction-dataset.custody-prediction-data-preparation.lib.filepath_process'))
get_path_from_json = fp.get_path_from_json

path = get_path_from_json('for_classifier_training/judgment_result_onehot_train_murphy', 'msg')
df_train = pd.read_msgpack(path)
# df_train = pd.read_msgpack('/Users/juanmurphy/Documents/ENV/custody-prediction-modeling-murphy/dataset/for_classifier_training/judgment_result_onehot_train_murphy(624).msg')

path = get_path_from_json('for_classifier_training/judgment_result_train_neu', 'msg')
df_train_neu = pd.read_msgpack(path)

path = get_path_from_json('for_classifier_training/judgment_result_onehot_test_murphy', 'msg')
df_test = pd.read_msgpack(path)
# df_test = pd.read_msgpack('/Users/juanmurphy/Documents/ENV/custody-prediction-modeling-murphy/dataset/for_classifier_training/judgment_result_onehot_test_murphy(156).msg')

path = get_path_from_json('for_classifier_training/judgment_result_test_neu', 'msg')
df_test_neu = pd.read_msgpack(path)

# 將 neutral 與 非neutral 分開
import matplotlib
categorical = ['Result','Willingness','AK','RK','AN','RN', 'Type']
meta_info = ['filename', 'ID', 'Others']
# column_prefixes = ['AA', 'RA', 'AD', 'RD', 'neutral']
df_list = [df_train, df_test]
df_list_neu = [df_train_neu, df_test_neu]

for df, df2 in zip(df_list,df_list_neu):
    all_neutral_columns = df2.columns[df2.columns.to_series().str.contains('neutral')].tolist()

    non_neutral_columns = sorted(list(set(list(matplotlib.cbook.flatten(df.columns.tolist()))) - set(all_neutral_columns) - set(categorical) - set(meta_info)))

# 分詞+清理
# 處理理據句

meta_info = ['filename', 'ID', 'Others']
categorical = ['Result','Willingness','Type', 'AK','RK','AN','RN']

df_list2 = []

for df in df_list:
    df2 = pd.DataFrame(columns=df.columns)
    df2[meta_info+categorical] = df[meta_info+categorical]
    df_list2.append(df2)

from itertools import count
from importlib import import_module
dp = import_module('custody-prediction-dataset.custody-prediction-data-preparation.lib.data_preprocess')
txt_to_clean = dp.txt_to_clean
clean_to_seg = dp.clean_to_seg
# set True for debug
debug = False
for index, df, df2 in zip(count(), df_list, df_list2):
    for i_column in non_neutral_columns:
        # select none empty entries, and apply txt_to_clean, clean_to_seg, seg_to_DocVec
        if debug:
            df2[i_column] = df[i_column].apply(lambda x: i_column)
        else:
            df2[i_column] = df.loc[~df.loc[:,i_column].isnull()][i_column].apply(txt_to_clean, textmode=True).apply(clean_to_seg, textmode=True)

# 處理中性句

df_list2_neu = []

for df in df_list_neu:
    df2 = pd.DataFrame(columns=df.columns)
    df2['ID'] = df['ID']
    df_list2_neu.append(df2)

from itertools import count
from importlib import import_module
dp = import_module('custody-prediction-dataset.custody-prediction-data-preparation.lib.data_preprocess')
txt_to_clean = dp.txt_to_clean
clean_to_seg = dp.clean_to_seg
# set True for debug
debug = False
for index, df, df2 in zip(count(), df_list_neu, df_list2_neu):
    for i_column in all_neutral_columns:
        # select none empty entries, and apply txt_to_clean, clean_to_seg, seg_to_DocVec
        if debug:
            df[i_column] = df[i_column].apply(lambda x: np.nan if type(x)==int else x)
            df2[i_column] = df[i_column].apply(lambda x: print(x) if type(x) != str else None)
        else:
            df[i_column] = df[i_column].apply(lambda x: np.nan if type(x)==int else x)
            df2[i_column] = df.loc[~df.loc[:,i_column].isnull()][i_column].apply(txt_to_clean, textmode=True).apply(clean_to_seg, textmode=True)

df_train = df_list2[0]
df_test = df_list2[1]
df_train_neu = df_list2_neu[0]
df_test_neu = df_list2_neu[1]
from importlib import import_module, reload
fp = reload(import_module('custody-prediction-dataset.custody-prediction-data-preparation.lib.filepath_process'))
save_path_to_json = fp.save_path_to_json
# save training set
total_number = len(df_train)
path = save_path_to_json('for_classifier_training/judgment_result_seg_train', 'msg', str(total_number))
df_train.to_msgpack(path)

path = save_path_to_json('for_classifier_training/judgment_result_seg_train_neu', 'msg', str(total_number))
df_train_neu.to_msgpack(path)

# save testing set
total_number = len(df_test)
path = save_path_to_json('for_classifier_training/judgment_result_seg_test', 'msg', str(total_number))
df_test.to_msgpack(path)

path = save_path_to_json('for_classifier_training/judgment_result_seg_test_neu', 'msg', str(total_number))
df_test_neu.to_msgpack(path)





