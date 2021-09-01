# %%
#//TODO: 弘祥註解
import pandas as pd
import numpy as np
advantage_list = pd.read_csv("./data/cleaned/sentence_advantage.csv")['0'].tolist()
disadvantage_list = pd.read_csv("./data/cleaned/sentence_disadvantage.csv")['0'].tolist()
neutral_list = pd.read_csv("./data/cleaned/sentence_neutral.csv")['0'].tolist()
# %%
# 不利標為0
y0 = [0]*len(disadvantage_list)
# 有利標為1
y1 = [1]*len(advantage_list)
# 中性標為2
y2 = [2]*len(neutral_list)

y = y0+y1+y2
X = disadvantage_list + advantage_list + neutral_list


df = pd.DataFrame({'y':y,'X':X})
# %%
import data_preprocess as dp
txt_to_clean = dp.txt_to_clean
clean_to_seg = dp.clean_to_seg
df['X_clean'] = df['X'].apply(txt_to_clean, textmode=True)
df['X_seg'] = df['X_clean'].apply(clean_to_seg, textmode=True)
# %%
from sklearn.utils import shuffle

np.random.seed(1234)

X, y = shuffle(df['X_seg'], df['y'])

X.reset_index(drop=True, inplace=True)
y.reset_index(drop=True, inplace=True)
# %%
from gensim.models.doc2vec import TaggedDocument
import collections

# label 維度
label_num = len(collections.Counter(y))

# add tags to documents
# 2 tags: 1. doc index 'SENT'+str(i)  2. labels '0', '1'
tagged_X_train = [TaggedDocument(words=X[i], tags=['SENT'+str(i), 'LABEL'+str(y[i])]) for i in range(len(X))]
# %%
from gensim.models.doc2vec import Doc2Vec
import multiprocessing

cores = multiprocessing.cpu_count()

model_dbow = Doc2Vec(dm=0, dm_mean=1, vector_size=100, window=10, negative=5, hs=0, min_count=2, sample=0, workers=cores, alpha=0.065, min_alpha=0.005)
# update vocabulary
model_dbow.build_vocab(tagged_X_train)
# %%
model_dmm = Doc2Vec(dm=1, dm_mean=1, vector_size=100, window=10, negative=5, hs=0, min_count=2, sample=0, workers=cores, alpha=0.065, min_alpha=0.005)
# update vocabulary
model_dmm.build_vocab(tagged_X_train)
# %%
import doc2vec_helper as dh
dh.model_train(model_dbow, tagged_X_train, alpha=0.05, min_alpha=0.05, max_epochs=71)

# %%
import random
random.seed(1234)
dh.sample_most_least_similar(model_dbow, tagged_X_train, X, y, n_sample=5)

# %%
model_dbow.save('./data/model/dbow_100_ADV_DIS_model.bin')
model_dmm.save('./data/model/dmm_100_ADV_DIS_model.bin')