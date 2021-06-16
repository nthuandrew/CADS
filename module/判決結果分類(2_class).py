# 判決結果分類(2_class)
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.utils import shuffle

from importlib import import_module, reload
fp = reload(import_module('dataset.custody-prediction-dataset.custody-prediction-data-preparation.lib.filepath_process'))
get_path_from_json = fp.get_path_from_json

path = get_path_from_json('dataset/for_classifier_training/judgment_result_doc2vec_train', 'msg')
df_train = pd.read_msgpack(path)

path = get_path_from_json('dataset/for_classifier_training/judgment_result_doc2vec_val', 'msg')
df_val = pd.read_msgpack(path)
# df_val = pd.read_msgpack('/Users/juanmurphy/Documents/ENV/custody-prediction-modeling-murphy/dataset/for_classifier_training/judgment_result_doc2vec_val(125).msg')

path = get_path_from_json('dataset/for_classifier_training/judgment_result_doc2vec_test', 'msg')
df_test = pd.read_msgpack(path)
# df_test = pd.read_msgpack('/Users/juanmurphy/Documents/ENV/custody-prediction-modeling-murphy/dataset/for_classifier_training/judgment_result_doc2vec_test(156).msg')

path = get_path_from_json('dataset/for_classifier_training/judgment_result_avg_neutral_vector_train', 'msg')
df_neutral_train = pd.read_msgpack(path)

path = get_path_from_json('dataset/for_classifier_training/judgment_result_avg_neutral_vector_val', 'msg')
df_neutral_val = pd.read_msgpack(path)
df_neutral_test = df_neutral_val

# remove unnecessary neutral vector
df_train = df_train[df_train.columns[~df_train.columns.str.contains('neutral')]]
df_val = df_val[df_val.columns[~df_val.columns.str.contains('neutral')]]
df_test = df_test[df_test.columns[~df_test.columns.str.contains('neutral')]]

df_list = [df_train, df_val, df_test]

path = get_path_from_json('dataset/for_classifier_training/judgment_result_avg_neutral_vector_train', 'msg')
df_neutral_train = pd.read_msgpack(path)

path = get_path_from_json('dataset/for_classifier_training/judgment_result_avg_neutral_vector_val', 'msg')
df_neutral_val = pd.read_msgpack(path)
df_neutral_test = df_neutral_val

# #### 取出 Result == [0,1] or [1,0] 的案例

for df in df_list:
    df['Result'] = df['Result'].apply(lambda x: x[:2])
    drop_list = df[['Result']].apply(lambda x: x.name if np.array_equal(x['Result'], np.array([0,0])) else None , axis=1).dropna().tolist()
    df.drop(index=drop_list, inplace=True)
    df.reset_index(drop=True, inplace=True)

# 紀錄資料增強前的 Result
for df in df_list:
    df['Result_ori'] = df['Result']
# Debug
# 資料個數
# number of class
# for df in df_list:
#     print(Counter(np.array(df['Result'].tolist()).argmax(axis=1)))
# print('train:val:test = %d:%d:%d' % (len(df_train),len(df_val),len(df_test)))

# Data Augmentation
from collections import defaultdict
# in windows, you should assign dtype to avoid it automatically fallback to int32
# which inconsistance with dataframe's int64
swap_result_map = {np.array([1,0], dtype=df_train['Result_ori'][0].dtype).tobytes():np.array([0,1]),np.array([0,1], dtype=df_train['Result_ori'][0].dtype).tobytes():np.array([1,0])}
def swapReasoning(x, selected_list=None):
#     print(x)
#     print(x['AA_concat'])

    if selected_list is not None:
        if x.name in selected_list:
            # swap AA RA
            tmp_array = x['AA_concat']
            x['AA_concat'] = x['RA_concat']
            x['RA_concat'] = tmp_array

            # swap AD RD
            tmp_array = x['AD_concat']
            x['AD_concat'] = x['RD_concat']
            x['RD_concat'] = tmp_array

            # swap result
            # print("origin", x['Result_ori'])
            x['Result'] = swap_result_map[x['Result_ori'].tobytes()]
            # print("swap", x['Result'])
            
            x['swap'] = True
        else:
            x['swap'] = False
            
    return x

def sentenceMean(x):
    def removeNan(items_list):
        return [item for item in items_list if ~np.isnan(item).any()]
    
    # take mean
    temp_list = removeNan(x.tolist())
    y = np.nanmean(temp_list, axis=0)
    return y
    
    
def getReasoningVector(df, df_neutral, column_prefix, new_prefix):
    column_list = df.columns[df.columns.str.contains(column_prefix + '(?!_neu|_concat)')]
#     df[new_prefix] = df[column_list].apply(sentenceMean, axis=1)
    # in windows, you need to pack whole ndarray as a list object in one column, 
    # then unlist it to get the ndarray into one column 
    # otherwise, you'll meet inconsist shape error 
    df_tmp = df[column_list].apply(lambda x: [sentenceMean(x)], axis=1)
    df_tmp = df_tmp.apply(lambda x: x[0])
    df[new_prefix] = df_tmp
    
    
def getNeutralVector(df, df_neutral, column_prefix, new_prefix):
    multiple_times = int(len(df) / len(df_neutral)) + 1
    df[new_prefix] = shuffle(pd.concat([df_neutral['avg_neutral_vector']]*multiple_times), random_state=5678)                                         .reset_index(drop=True).iloc[0:len(df)].apply(lambda x: x)
    
def getAugmentedVector(df, df_neutral, column_prefix, new_prefix):
#     df[new_prefix] = df[[column_prefix+'_neu_concat', column_prefix+'_concat']].apply(sentenceMean, axis=1)
    # in windows, you need to pack whole ndarray as a list object in one column, 
    # then unlist it to get the ndarray into one column 
    # otherwise, you'll meet inconsist shape error 
    df_tmp = df[[column_prefix+'_neu_concat', column_prefix+'_concat']].apply(lambda x: [sentenceMean(x)], axis=1)
    df_tmp = df_tmp.apply(lambda x: x[0])
    df[new_prefix] = df_tmp
    
    
def getConcatReasoningNeutralVector(df, df_neutral, column_prefix, new_prefix):
#     df[new_prefix] = df[column_prefix].apply(lambda x: np.concatenate(x), axis=1)
    # in windows, you need to pack whole ndarray as a list object in one column, 
    # then unlist it to get the ndarray into one column 
    # otherwise, you'll meet inconsist shape error 
    df_tmp = df[column_prefix].apply(lambda x: [np.concatenate(x)], axis=1)
    df_tmp = df_tmp.apply(lambda x: x[0])
    df[new_prefix] = df_tmp
    
    
def checkAugmentedVector(df, df_neutral, column_prefix, new_prefix):
    
    def removeNan(items_list):
        return [item for item in items_list if ~np.isnan(item).any()]
        
    df[[column_prefix+'_augmented', column_prefix+'_concat', column_prefix+'_neu_concat']].apply(
        lambda x: [True, x.name] if np.array_equal(
            x[column_prefix+'_augmented'], 
            np.nanmean(removeNan([x[column_prefix+'_concat'], x[column_prefix+'_neu_concat']]), axis=0)
            ) else [False, x.name], axis=1).apply(
        lambda x: print("fail index:", x[1]) if x[0] != True else None)
        
        
def loopForAll(df, df_neutral, prefixes_dict, column_prefixes_list, prefix, func):
    # loop for all (AA,AD,RA,RD)
    for column_prefix in column_prefixes_list[:4]:
        print(column_prefix)
        new_prefix = column_prefix + prefix
        print(new_prefix)
        
        func(df, df_neutral, column_prefix, new_prefix)
        #display(df)

        if prefixes_dict is not None:
            prefixes_dict[column_prefix].append(new_prefix)
        
        #print(prefixes_dict)

import matplotlib

categorical = ['Result','Willingness','AK','RK','AN','RN', 'Type']
meta_info = ['filename', 'ID', 'Others']
column_prefixes = ['AA', 'RA', 'AD', 'RD']

df_name_list = ['train', 'val', 'test']
df_dict = {'train':df_train, 'val':df_val, 'test':df_test}
df_neutral_dict = {'train':df_neutral_train, 'val':df_neutral_val, 'test':df_neutral_test}

# np.random.seed(1234)

swap_reasoning_rate = 0.50
identical_reasoning_rate = 0.1
drop_reasoning_rate = 0.1


random_selected_row_index_dict = defaultdict(defaultdict)

for name in df_name_list:
    # print(name)
    tmp_dict = defaultdict(list)
    total_list = range(len(df_dict[name]))
    
    # select swap reasoning cases
    #partial_list = shuffle(total_list[selected_length:], random_state=5566)
    partial_list = shuffle(total_list, random_state=5566)
    selected_length = int(len(partial_list)*swap_reasoning_rate)
    swap_reasoning_list = partial_list[0:selected_length]
    tmp_dict['swap_reasoning'] = swap_reasoning_list 
    random_selected_row_index_dict[name] = tmp_dict
    
mean_prefixes = defaultdict(list)

neu_mean_prefixes = defaultdict(list)
# take mean of reasoning sentence (AA,AD,RA,RD)
loopForAll(df_dict['train'], df_neutral=None, prefixes_dict=mean_prefixes, 
           column_prefixes_list=column_prefixes, prefix='_concat', func=getReasoningVector)

import matplotlib

categorical = ['Result','Willingness','AK','RK','AN','RN', 'Type']
meta_info = ['filename', 'ID', 'Others']
column_prefixes = ['AA', 'RA', 'AD', 'RD']

df_name_list = ['train', 'val', 'test']
df_dict = {'train':df_train, 'val':df_val, 'test':df_test}
df_neutral_dict = {'train':df_neutral_train, 'val':df_neutral_val, 'test':df_neutral_test}

# np.random.seed(1234)

swap_reasoning_rate = 0.50
identical_reasoning_rate = 0.1
drop_reasoning_rate = 0.1


random_selected_row_index_dict = defaultdict(defaultdict)

for name in df_name_list:
    # print(name)
    tmp_dict = defaultdict(list)
    total_list = range(len(df_dict[name]))
    
    # select swap reasoning cases
    #partial_list = shuffle(total_list[selected_length:], random_state=5566)
    partial_list = shuffle(total_list, random_state=5566)
    selected_length = int(len(partial_list)*swap_reasoning_rate)
    swap_reasoning_list = partial_list[0:selected_length]
    tmp_dict['swap_reasoning'] = swap_reasoning_list
    random_selected_row_index_dict[name] = tmp_dict


for name in df_name_list:
    print(name)
    print(len(df_dict[name]))

    mean_prefixes = defaultdict(list)

    # take mean of reasoning sentence (AA,AD,RA,RD)
    loopForAll(df_dict[name], df_neutral=None, prefixes_dict=mean_prefixes, 
               column_prefixes_list=column_prefixes, prefix='_concat', func=getReasoningVector)
    
    
    # take mean of neutral sentence (neutral...)
    neu_mean_prefixes = defaultdict(list)
    
    loopForAll(df_dict[name], df_neutral=df_neutral_dict[name], prefixes_dict=neu_mean_prefixes, 
               column_prefixes_list=column_prefixes, prefix='_neu_concat', func=getNeutralVector)
    
    if name == 'train':
        # random swap reasoning
        ## swap should before drop, because drop will modify x['Result'],
        ## but swap takes x['Result_ori'] as mapping key to modify x['Result'], 
        ## it'll overwrite drop's modify if swap after drop
        print("swap reasoning...")
        df_dict[name] = df_dict[name].apply(swapReasoning, selected_list=random_selected_row_index_dict[name]['swap_reasoning'], axis=1)

    # take mean of reasoning sentence and neutral sentence (AA,AD,RA,RD + neutral...)
    augmented_prefixes = defaultdict(list)
    
    loopForAll(df_dict[name], df_neutral=None, prefixes_dict=augmented_prefixes, 
               column_prefixes_list=column_prefixes, prefix='_augmented', func=getAugmentedVector)
    
    
    # check consistency of augmented vector
    loopForAll(df_dict[name], df_neutral=None, prefixes_dict=None, 
               column_prefixes_list=column_prefixes, prefix='_checked', func=checkAugmentedVector)

    # concatenate
    print("concatenating...")
    augmented_prefix_list = list(matplotlib.cbook.flatten(list(augmented_prefixes.values())))
    getConcatReasoningNeutralVector(df_dict[name], df_neutral=None, column_prefix=augmented_prefix_list, new_prefix='X')


# check array value
column_prefix='RD'
name='train'
index=3
(df_dict[name][df_dict[name].columns[
            df_dict[name].columns.str.contains(column_prefix)]])

item = (df_dict[name][df_dict[name].columns[
            df_dict[name].columns.str.contains(column_prefix)]])[column_prefix+'_concat'][index]

if ~np.isnan(item).any():
    print("\nlength is:")
    print(len(item))
(df_dict[name][df_dict[name].columns[
            df_dict[name].columns.str.contains(column_prefix)]])[column_prefix+'_neu_concat'][index]

print(item)

if ~np.isnan(item).any():
    print("\nlength is:")
    print(len(item))
print(df_dict['train'].iloc[5].to_string())

mean_prefixes_list = list(matplotlib.cbook.flatten(list(mean_prefixes.values())))

for name in df_name_list:
    print(name)
    if name == 'train':
        #display(df_dict[name][mean_prefixes_list+['swap', 'identical', 'drop']])
        display(df_dict[name][mean_prefixes_list+['swap']])
    else:
        display(df_dict[name][mean_prefixes_list])

neu_prefixes_list = list(matplotlib.cbook.flatten(list(neu_mean_prefixes.values())))

for name in df_name_list:
    print(name)
    display(df_dict[name][neu_prefixes_list])

augmented_prefix_list = list(matplotlib.cbook.flatten(list(augmented_prefixes.values())))

for name in df_name_list:
    print(name)
    display(df_dict[name][augmented_prefix_list])

# 檢查 vector 長度
docvec_size = []

for i_column in augmented_prefix_list:
    docvec_size.append(len(df_dict['train'][i_column].loc[1]))

print(docvec_size)

# prepare X, y

from sklearn.model_selection import train_test_split

X_train_list = []
X_val_list = []
X_test_list = []

X_dict_of_list = {'train':X_train_list, 'val':X_val_list, 'test':X_test_list}

y_train_list = []
y_val_list = []
y_test_list = []

y_dict_of_list = {'train':y_train_list, 'val':y_val_list, 'test':y_test_list}


for name in df_name_list:
    for df_tmp in [df_dict[name]]:
        X, y = np.array(df_tmp['X'].apply(lambda x: x.tolist()).tolist()), np.array(df_tmp['Result'].apply(lambda x: x.tolist()).tolist())

        X_dict_of_list[name].append(X)
        y_dict_of_list[name].append(y)
        
for i in range(len(X_dict_of_list['train'])):
    print('multiple:', i+1)
    print('train:val:test = %d:%d:%d' % (len(X_dict_of_list['train'][i]),len(X_dict_of_list['val'][i]),len(X_dict_of_list['test'][i])))
    print('train:val:test = %d:%d:%d' % (len(y_dict_of_list['train'][i]),len(y_dict_of_list['val'][i]),len(y_dict_of_list['test'][i])))
    #print('train:val:test = %d:%d:%d' % (len(y_train_list[i]),len(y_val_list[i]),len(y_test_list[i])))
    
for i in range(len(X_dict_of_list['train'])):
    print('multiple:', i+1)

    # number of instance by class
    print('number of instance by class(train):', y_dict_of_list['train'][i].sum(axis=0))
    print('number of instance by class(val):', y_dict_of_list['val'][i].sum(axis=0))
    print('number of instance by class(test):', y_dict_of_list['test'][i].sum(axis=0))

for i in range(len(X_dict_of_list['train'])):
    print('multiple:', i+1)
    print('X_train.shape', X_dict_of_list['train'][i].shape)
    print('y_train.shape', y_dict_of_list['train'][i].shape)

# 檢查是否有非 one-hot 項
for name in df_name_list:
    print(name)
    for i in range(len(X_dict_of_list[name])):
        print('multiple:', i+1)
        # search for non one-hot items
        pd.DataFrame(y_dict_of_list[name][i]).apply(lambda x: print(x) if x.sum() != 1 else None, axis=1)


# ### oversampling -- resample
# 
# - [Comparison of the different over-sampling algorithms — imbalanced-learn 0.4.3 documentation](http://imbalanced-learn.org/en/stable/auto_examples/over-sampling/plot_comparison_over_sampling.html#sphx-glr-auto-examples-over-sampling-plot-comparison-over-sampling-py)
# 
# - [Comparison of the different under-sampling algorithms — imbalanced-learn 0.4.3 documentation](http://imbalanced-learn.org/en/stable/auto_examples/under-sampling/plot_comparison_under_sampling.html#sphx-glr-auto-examples-under-sampling-plot-comparison-under-sampling-py)

from imblearn.over_sampling import RandomOverSampler # doctest: +NORMALIZE_WHITESPACE
from sklearn.utils import shuffle
from collections import Counter

X_train_res_list = []
X_dict_of_list['train_res'] = X_train_res_list

y_train_res_list = []
y_dict_of_list['train_res'] = y_train_res_list

for i in range(len(X_dict_of_list['train'])):

    ros = RandomOverSampler(sampling_strategy='auto',random_state=42)
    X_train_res, y_train_res = ros.fit_resample(X_dict_of_list['train'][i], y_dict_of_list['train'][i])
    
    X_train_res, y_train_res = shuffle(X_train_res, y_train_res, random_state=1234)
    
    # in 2 classes case, it'll return index, so you must turn into onehot encode...
    tmp_array = np.zeros(shape=(len(y_train_res), y_dict_of_list['train'][i].shape[1]))
    for j in range(len(tmp_array)):
        tmp_array[j][y_train_res[j][0]] = 1
    

    X_dict_of_list['train_res'].append(X_train_res)
    y_dict_of_list['train_res'].append(tmp_array)
    
    print("before resample:", Counter(y_dict_of_list['train'][i].argmax(axis=1)))
    print("after resample:", Counter(y_dict_of_list['train_res'][i].argmax(axis=1)))

# 建立Model
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.disable_eager_execution()

from importlib import reload
import lib.NN_models

lib.NN_models = reload(lib.NN_models)
from lib.NN_models import splited_DNN
feed_dict = {}
feed_dict['x'] = X_dict_of_list['train_res'][0][:]
feed_dict['y'] = y_dict_of_list['train_res'][0][:]
feed_dict['k'] = 1.0
model = splited_DNN(feed_dict['x'], feed_dict['y'], num_or_size_splits=docvec_size, bottleneck_size=400)

model.print_cost(feed_dict=feed_dict)

# Training
import time
import random
import sys
from pathlib import Path, PureWindowsPath

test_list = [256]
for number in test_list:

    for i in range(1):
        print("loop: ", i)

        import lib.score_function
        from importlib import reload
        reload(lib.score_function)

        from lib.score_function import print_score


        model_list = []

        # epoch = random.randrange(5, 80)
        epoch =30

        for i in range(0,len(X_dict_of_list['train']),5):
        #     print('multiple:', i+1)

            seed = 1234
            # fixed random seed (help reproduce)
    #         seed = 6887378808282378165
            print("Seed was:", seed)

            class_weights=[1.0, 1.0]

            model = splited_DNN(X_dict_of_list['train_res'][i], y_dict_of_list['train_res'][i], num_or_size_splits=docvec_size, bottleneck_size=number, class_weights=class_weights, seed=seed)
            valid_accuracy = model.train(X_dict_of_list['train_res'][i], y_dict_of_list['train_res'][i], X_dict_of_list['val'][i], y_dict_of_list['val'][i], epoch=epoch)
            print("test with data augmentation: ")
            accuracy, roc_auc, pr_auc, f1 = print_score(model, X_dict_of_list['test'][i], y_dict_of_list['test'][i], logging=None)
            print('accuracy', accuracy)
            print('roc_auc', roc_auc)
            print('pr_auc', pr_auc)
            print('f1', f1)
            print('\n')
        #     print("test without data augmentation: ")
        #     print_score(model, X_dict_of_list['test'][0], y_dict_of_list['test'][0], show_threshold=False)
        #     print('\n')

            current_time = time.strftime("%Y%m%d%H%M%S", time.localtime())

    #         roc_auc_str = {k:round(v,2) if isinstance(v,float) else v for k,v in roc_auc.items()}
    #         roc_auc_str = ''.join('{}-{}_'.format(key, val) for key, val in roc_auc_str.items())
            roc_auc_str = round(roc_auc, 2)
            #"roc_auc_{}.ckpt".format(roc_auc_str)

            pr_auc_str = round(pr_auc, 2)
            f1_str = round(f1, 2)

            class_weights_str = {k:round(v,2) if isinstance(v,float) else v for k,v in enumerate(class_weights)}
            class_weights_str = ''.join('{}-{}_'.format(key, val) for key, val in class_weights_str.items())

            class_num = y_dict_of_list['train'][i].shape[1]

            modelFilePath = ("./models/SplitDNN判決結果分類/"
                "SplitDNN({}_class)"
                "_train{}_val{}_test{}"
                "_class_weights{}_epoch{}"
                "_valid_accuracy{:.4f}_accuracy{:.4f}"
                "_roc_auc{}_seed{}_{}.ckpt").format(
                    class_num,
                    len(X_dict_of_list['train'][i]),
                    len(X_dict_of_list['val'][i]),
                    len(X_dict_of_list['test'][i]),
                    class_weights_str,
                    epoch,
                    valid_accuracy, 
                    accuracy,
                    roc_auc_str,
                    pr_auc_str,
                    f1_str,
                    seed, 
                    current_time)

            modelFilePath = Path(modelFilePath)
            modelFilePath = Path(modelFilePath.absolute())
            # to avoid model.save() crached by filename too long issue under windows, 
            # use absolute path prefixed with u'\\\\?\\' with PathLib, 
            # then convert it to string as model.save()'s path argument.
            unc_prefix = PureWindowsPath(u'\\\\?\\')
            unc_prefix = Path(unc_prefix)
            unc_modelFilePath = Path(str(unc_prefix) + str(modelFilePath))

            if unc_modelFilePath.parents[0].exists():
                print(str(unc_modelFilePath))
                model.save(str(unc_modelFilePath))
            else:
                print(str(modelFilePath))
                model.save(str(modelFilePath))


            # model.load only accept normal short path, not unc path,
            # so just output latest.ckpt as short filename
            latest_path = str((unc_modelFilePath.parents[0])/'latest.ckpt').replace(str(unc_prefix), '')
            print(latest_path)

            model.save(latest_path)
    
accuracy, roc_auc, pr_auc, f1 = print_score(model, X_dict_of_list['test'][0], y_dict_of_list['test'][0])


# load model
import numpy as np
import tensorflow as tf

from importlib import reload
import lib.NN_models

lib.NN_models = reload(lib.NN_models)
from lib.NN_models import splited_DNN

# model = splited_DNN(X_dict_of_list['train_res'][0], y_dict_of_list['train_res'][0], num_or_size_splits=docvec_size, bottleneck_size=60)
model = splited_DNN(np.zeros(shape=(1,800)), np.zeros(shape=(1,2)), num_or_size_splits=[200, 200, 200, 200], bottleneck_size=60)
model.load(latest_path)