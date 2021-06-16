# 訓練有利不利中性文字Doc2Vec

# 讀取標注資料
import pandas as pd
import numpy as np
from importlib import import_module, reload
fp = reload(import_module('dataset.custody-prediction-dataset.custody-prediction-data-preparation.lib.filepath_process'))
get_path_from_json = fp.get_path_from_json
path = get_path_from_json('for_feature_extraction/sentence_advantages_train_murphy', 'msg')
# path = '/Users/juanmurphy/Documents/ENV/custody-prediction-modeling-murphy/dataset/for_feature_extraction/sentence_advantages_train_gitlab(3692).msg'
advantage_list = pd.read_msgpack(path)[0].tolist()
path = get_path_from_json('for_feature_extraction/sentence_disadvantages_train_murphy', 'msg')
# path = '/Users/juanmurphy/Documents/ENV/custody-prediction-modeling-murphy/dataset/for_feature_extraction/sentence_disadvantages_train_gitlab(1289).msg'
disadvantage_list = pd.read_msgpack(path)[0].tolist()
path = get_path_from_json('for_feature_extraction/sentence_neutrals_train_murphy', 'msg')
# path = '/Users/juanmurphy/Documents/ENV/custody-prediction-modeling-murphy/dataset/for_feature_extraction/sentence_neutrals_train_gitlab(2490).msg'
neutral_list = pd.read_msgpack(path)[0].tolist()
# 準備訓練資料

# 加上標註(不利,有利,中性=0,1,2)
# 不利標為0
y0 = [0]*len(disadvantage_list)
# 有利標為1
y1 = [1]*len(advantage_list)
# 中性標為2
y2 = [2]*len(neutral_list)
y = y0+y1+y2
X = disadvantage_list + advantage_list + neutral_list
df = pd.DataFrame({'y':y,'X':X})
# 清理+斷詞
from importlib import import_module
dp = import_module('dataset.custody-prediction-dataset.custody-prediction-data-preparation.lib.data_preprocess')
txt_to_clean = dp.txt_to_clean
clean_to_seg = dp.clean_to_seg
df['X_clean'] = df['X'].apply(txt_to_clean, textmode=True)
df['X_seg'] = df['X_clean'].apply(clean_to_seg, textmode=True)
# 打亂順序
from sklearn.utils import shuffle
np.random.seed(1234)
X, y = shuffle(df['X_seg'], df['y'])
X.reset_index(drop=True, inplace=True)
y.reset_index(drop=True, inplace=True)
# ## 訓練 Doc2Vec
# 
# - [gensim: models.doc2vec – Doc2vec paragraph embeddings](https://radimrehurek.com/gensim/models/doc2vec.html)
# 
# - [Multi-Class Text Classification with Doc2Vec & Logistic Regression](https://towardsdatascience.com/multi-class-text-classification-with-doc2vec-logistic-regression-9da9947b43f4)
# 
# - [A gentle introduction to Doc2Vec – ScaleAbout – Medium](https://medium.com/scaleabout/a-gentle-introduction-to-doc2vec-db3e8c0cce5e)
# 
# > we will use **gensim** [**implementation**](https://rare-technologies.com/doc2vec-tutorial/)of **doc2vec.** here is how the gensim TaggedDocument object looks like:
# > 
# > ![](https://cdn-images-1.medium.com/max/1200/1*As22mK8YKolvVFCGmHvGxw.png)
# 
# > and then we can check the similarity of every unique **document** to every **tag**, this way:
# > 
# > ![](https://cdn-images-1.medium.com/max/1200/1*T9swFeb7vqTOWKA9NNnIDA.png)
# ### 初始化

from gensim.models.doc2vec import TaggedDocument
import collections
# label 維度
label_num = len(collections.Counter(y))

# add tags to documents
# 2 tags: 1. doc index 'SENT'+str(i)  2. labels '0', '1'
tagged_X_train = [TaggedDocument(words=X[i], tags=['SENT'+str(i), 'LABEL'+str(y[i])]) for i in range(len(X))]
from gensim.models.doc2vec import Doc2Vec
import multiprocessing

cores = multiprocessing.cpu_count()

model_dbow = Doc2Vec(dm=0, dm_mean=1, vector_size=100, window=10, negative=5, hs=0, min_count=2, sample=0, workers=cores, alpha=0.065, min_alpha=0.005)
# update vocabulary
model_dbow.build_vocab(tagged_X_train)

model_dmm = Doc2Vec(dm=1, dm_mean=1, vector_size=100, window=10, negative=5, hs=0, min_count=2, sample=0, workers=cores, alpha=0.065, min_alpha=0.005)
# update vocabulary
model_dmm.build_vocab(tagged_X_train)


# ### 訓練
# 
# - [DOC2VEC gensim tutorial – Deepak Mishra – Medium](https://medium.com/@mishra.thedeepak/doc2vec-simple-implementation-example-df2afbbfbad5)
# 
#     Note: dm defines the training algorithm. If dm=1 means ‘distributed memory’ (PV-DM) and dm =0 means ‘distributed bag of words’ (PV-DBOW). Distributed Memory model preserves the word order in a document whereas Distributed Bag of words just uses the bag of words approach, which doesn’t preserve any word order.
#     
# - [build_vocab fails when calling with different trim_rule for same corpus · Issue #1187 · RaRe-Technologies/gensim](https://github.com/RaRe-Technologies/gensim/issues/1187)
# 
#     In general, triggering `build_vocab()` more than once, without the (un my opinion experimental/sketchy) `update` parameter, isn't a supported/well-defined operation.
#     
# #### 相似性檢查
# 
# - [gensim/doc2vec-lee.ipynb at develop · RaRe-Technologies/gensim](https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-lee.ipynb)
# 
# 用訓練出來的 model 對 training set 的文件重新計算一次vector，用這 vector 再去跟 model 訓練出來的每個 doc_id 對應的 doc_vector 計算相似度，取出相似度最高的前10名的doc_id。排名第一的 doc_id 應該要跟原本training set 文件的 doc_id 是一致的，只有少部份不一致要到排名第二的doc_id 才會跟原文件一致。
import lib.Doc2Vec_helper
from lib.Doc2Vec_helper import model_train, sanity_check, show_most_least_similar, sample_most_least_similar

from importlib import reload
reload(lib.Doc2Vec_helper)

model_train(model_dbow, tagged_X_train, alpha=0.05, min_alpha=0.05, max_epochs=71)

model_train(model_dmm, tagged_X_train, alpha=0.05, min_alpha=0.05, max_epochs=71)
# #顯示最相近與最不相近句子

import random
random.seed(1234)
sample_most_least_similar(model_dbow, tagged_X_train, X, y, n_sample=5)
random.seed(8901)
sample_most_least_similar(model_dmm, tagged_X_train, X, y, n_sample=6)

# 儲存模型
model_dbow.save('models/Doc2Vec_有利不利/dbow_100_ADV_DIS_model.bin')

model_dmm.save('models/Doc2Vec_有利不利/dmm_100_ADV_DIS_model.bin')

import pickle
# save tagged sentence to binary
pickle.dump(tagged_X_train, open("models/Doc2Vec_有利不利/tagged_sentence.pickle", "wb"))

# load tagged sentence from binary
tagged_X_train = pickle.load(open("models/Doc2Vec_有利不利/tagged_sentence.pickle", "rb"))