# 判決結果分類_產生平均中性向量
# 讀取標注資料
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.utils import shuffle

from importlib import import_module, reload
fp = reload(import_module('dataset.custody-prediction-dataset.custody-prediction-data-preparation.lib.filepath_process'))
get_path_from_json = fp.get_path_from_json

path = get_path_from_json('dataset/for_classifier_training/judgment_result_doc2vec_train_neu', 'msg')
df_train = pd.read_msgpack(path)
# df_train = pd.read_msgpack('/Users/juanmurphy/Documents/ENV/custody-prediction-modeling/dataset/for_classifier_training/judgment_result_doc2vec_train_neu(380).msg')

path = get_path_from_json('dataset/for_classifier_training/judgment_result_doc2vec_val_neu', 'msg')
df_val = pd.read_msgpack(path)
# df_test = pd.read_msgpack('./dataset/judgment_result_doc2vec_test(157).msg')

all_neutral_columns = df_train.columns[df_train.columns.to_series().str.contains('neutral')].tolist()
neu_vector_train_list = df_train[all_neutral_columns].stack().tolist()
neu_vector_val_list = df_val[all_neutral_columns].stack().tolist()

# 產生平均中性向量
import time

def getAvgVector(data_vector_list, data_index, verbose=False):

    temp_vector_list = []

    max_loop_count = 15000
    counter = 0
    interval = 20
    max_counter = 10

    previous_avg_vector = data_vector_list[0]
    previous_avg_vector2 = data_vector_list[0]

    startTime = time.time()
    
    for i, vector in enumerate(data_vector_list[1:]):
        temp_vector_list.append(vector)
        
        if i >= max_loop_count:
            break

        if verbose:
            current_avg_vector2 = np.mean(temp_vector_list, axis=0)
            delta2 = np.linalg.norm(current_avg_vector2 - previous_avg_vector2)
            previous_avg_vector2 = current_avg_vector2
            print("loop {}: ".format(i), delta2)
            
        if i < 10 or i % interval == 0:
            current_avg_vector = np.mean(temp_vector_list, axis=0)
            delta = np.linalg.norm(current_avg_vector - previous_avg_vector)
            # judgement_status_by_id[random_id]['judgement_status']=current_y_pred_prob
            if verbose:
                print(i, delta)

            if delta < 1.0e-6:
                counter += 1
                interval = 1
                max_counter = 3
            elif delta < 0.001:
                counter += 1
                interval = 1
                max_counter = 10
            elif delta < 0.002:
                counter += 1
                #counter = 0
                interval = 2
                max_counter = 20
            elif delta < 0.005:
                counter += 1
                # counter = 0
                interval = 4
                max_counter = 40
            elif delta < 0.01:
                # counter += 1
                counter = 0
                interval = 10
                # max_counter = 80
            else:
                counter = 0
                interval = 20

            if counter >= max_counter:
                if verbose:
                    print(i, delta)
                break

            previous_avg_vector = current_avg_vector


    elapsTime = (time.time() - startTime)
    # Debug
    print("index:", data_index, "\telapsTime:", "{:.3f}".format(elapsTime), "\titer:", i, "\tdelta:", delta)
    
    return current_avg_vector


def getAvgVectorPool(data_vector_list, n_vector=100, verbose=False):
        
    avg_vector_pool = []

    #previous_avg_vector = getAvgVector(shuffle(neu_vector_list))

    startTime = time.time()
    for i in range(n_vector):
        avg_vector_pool.append(getAvgVector(shuffle(data_vector_list), data_index=i, verbose=verbose))

        # current_avg_vector = np.mean(neu_avg_pool, axis=0)
        # delta = np.linalg.norm(current_avg_vector - previous_avg_vector)
        #std_norm = np.linalg.norm(np.std(np.array(avg_vector_pool), axis=0))
        #print("std_norm:", std_norm)
        # previous_avg_vector = current_avg_vector
        
    elapsTime = (time.time() - startTime)
    
    std_norm = np.linalg.norm(np.std(np.array(avg_vector_pool), axis=0))
    # Debug
    print("index:", " ", "\telapsTime:", "{:.3f}".format(elapsTime), "\tstd_norm:", std_norm)

    return avg_vector_pool

neu_avg_train_pool = getAvgVectorPool(neu_vector_train_list, n_vector=1000)
neu_avg_val_pool = getAvgVectorPool(neu_vector_val_list, n_vector=1000)
# standard deviation of 100 norms of difference of neutral vector of training set and validation set 
np.std(np.linalg.norm((np.array(neu_avg_train_pool) - np.array(neu_avg_val_pool)), axis=1))
# 儲存中性向量

import pandas as pd
from importlib import import_module, reload
fp = reload(import_module('dataset.custody-prediction-dataset.custody-prediction-data-preparation.lib.filepath_process'))
save_path_to_json = fp.save_path_to_json

neu_avg_pool_dict = {'train':neu_avg_train_pool, 'val':neu_avg_val_pool}

for name in neu_avg_pool_dict.keys():
    neu_avg_df = pd.DataFrame(neu_avg_pool_dict[name])
    neu_avg_df2 = pd.DataFrame()

    # must have string object or it'll throw an error:
    # ValueError: cannot reshape array of size 200000 into shape (1,1000)
    neu_avg_df2['index'] = neu_avg_df.apply(lambda x: str(x.name), axis=1)
    neu_avg_df2['avg_neutral_vector'] = neu_avg_df.apply(lambda x: np.array(x), axis=1)

    path = save_path_to_json('dataset/for_classifier_training/judgment_result_avg_neutral_vector_{}'.format(name), 'msg', str(len(neu_avg_df)))
    neu_avg_df2.to_msgpack(path)

    result = pd.read_msgpack(path)

# 使用 pyarrow to_pybytes 寫到檔案中 
import pyarrow as pa
from importlib import import_module, reload
fp = reload(import_module('dataset.custody-prediction-dataset.custody-prediction-data-preparation.lib.filepath_process'))
save_path_to_json = fp.save_path_to_json
for name in neu_avg_pool_dict.keys():
    neu_avg_df = pd.DataFrame(neu_avg_pool_dict[name])
    neu_avg_df2 = pd.DataFrame()

    neu_avg_df2['index'] = neu_avg_df.apply(lambda x: str(x.name), axis=1)
    neu_avg_df2['avg_neutral_vector'] = neu_avg_df.apply(lambda x: np.array(x), axis=1)
    
    path = save_path_to_json('dataset/for_classifier_training/judgment_result_avg_neutral_vector_{}'.format(name), 'pyarrow', str(len(neu_avg_df)))
    
    with open(path, "wb") as file:
        pyarrow_dumped = pa.serialize(neu_avg_df2).to_buffer()
        file.write(pyarrow_dumped)

    with open(path, "rb") as file:
        pyarrow_dumped = file.read()

    result = pa.deserialize(pyarrow_dumped)